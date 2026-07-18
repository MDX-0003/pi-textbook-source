import {
  Agent,
  type AgentEvent,
  type AgentState,
} from "./agent.js";
import type { AgentRunResult } from "./agent-loop.js";
import { buildContext } from "./context.js";
import {
  formatResourceContext,
  type ExtensionHost,
  type ResourceCatalog,
} from "./resources.js";
import {
  pathTo,
  type MessageSessionEntry,
  type SessionEntry,
  type SessionStore,
} from "./session.js";
import {
  executeToolCall,
  type ToolExecutor,
  type ToolRegistry,
} from "./tool.js";
import {
  textOf,
  type AgentContext,
  type AgentMessage,
  type Model,
} from "./types.js";

export interface ContextBudget {
  total: number;
  reservedOutput: number;
  safetyMargin: number;
  estimateTokens(value: string | AgentMessage): number;
}

export interface RuntimeConfig {
  activeLeafId: string | null;
  systemPrompt?: string;
  maxSteps?: number;
  context: ContextBudget;
}

export interface RuntimeDeps {
  model: Model;
  tools: ToolRegistry;
  session: SessionStore;
  resources: ResourceCatalog;
  extensionHost?: ExtensionHost;
  createId(): string;
  now(): number;
}

export interface RuntimeControl {
  getState(): AgentState;
  subscribe(listener: (event: AgentEvent) => void): () => void;
  steer(value: string): void;
  followUp(value: string): void;
  abort(): void;
}

export interface Runtime {
  readonly control: RuntimeControl;
  readonly session: SessionStore;
  readonly resources: ResourceCatalog;
  readonly extensions: ExtensionHost | undefined;
  getActiveLeafId(): string | null;
  prompt(value: string): Promise<AgentRunResult>;
  flush(): Promise<void>;
  dispose(): Promise<void>;
}

export interface ContextProjectionSnapshot {
  activePath: readonly SessionEntry[];
  persistedMessageCount: number;
}

export type RuntimeMode = "interactive" | "print" | "json";

export interface ModeIO {
  write(value: string): void | Promise<void>;
}

function combinedSystemPrompt(
  configured: string | undefined,
  resources: ResourceCatalog,
): string | undefined {
  const resourceContext =
    resources.resources.length === 0
      ? undefined
      : formatResourceContext(resources, []);
  const parts = [configured, resourceContext]
    .filter(
      (value): value is string =>
        value !== undefined && value.trim().length > 0,
    );
  return parts.length === 0 ? undefined : parts.join("\n\n");
}

function messagesIn(
  entries: readonly SessionEntry[],
): AgentMessage[] {
  return entries.flatMap((entry) =>
    entry.type === "message"
      ? [structuredClone(entry.message)]
      : []
  );
}

function temporaryEntries(
  snapshot: ContextProjectionSnapshot,
  messages: readonly AgentMessage[],
): SessionEntry[] {
  if (
    !Number.isInteger(snapshot.persistedMessageCount) ||
    snapshot.persistedMessageCount < 0 ||
    snapshot.persistedMessageCount > messages.length
  ) {
    throw new Error("persistedMessageCount 超出了当前 model context");
  }

  const activePath = structuredClone(snapshot.activePath);
  const occupiedIds = new Set(activePath.map((entry) => entry.id));
  let parentId = activePath.at(-1)?.id ?? null;
  const suffix = messages.slice(snapshot.persistedMessageCount);
  const entries: MessageSessionEntry[] = suffix.map((message, index) => {
    let id = `__runtime_context_${index}`;
    while (occupiedIds.has(id)) id = `_${id}`;
    occupiedIds.add(id);
    const entry: MessageSessionEntry = {
      id,
      parentId,
      timestamp: message.timestamp,
      type: "message",
      message: structuredClone(message),
    };
    parentId = id;
    return entry;
  });
  return [...activePath, ...entries];
}

/**
 * Agent 保留完整 canonical transcript；这个 adapter 只在每次模型请求前，
 * 把尚未持久化的临时 suffix 接到当前 active path，再调用唯一 buildContext。
 */
export function createContextProjectingModel(
  inner: Model,
  config: Pick<RuntimeConfig, "systemPrompt" | "context">,
  getSnapshot: () => ContextProjectionSnapshot,
): Model {
  return {
    stream(context, options = {}) {
      const snapshot = getSnapshot();
      const projected = buildContext(
        temporaryEntries(snapshot, context.messages),
        {
          maxTokens: config.context.total,
          reservedOutput: config.context.reservedOutput,
          safetyMargin: config.context.safetyMargin,
          systemPrompt: config.systemPrompt,
          estimateTokens: config.context.estimateTokens,
        },
      );
      const request: AgentContext = {
        systemPrompt: projected.systemPrompt,
        messages: structuredClone(projected.messages),
        tools:
          context.tools === undefined
            ? undefined
            : structuredClone(context.tools),
      };
      return inner.stream(request, { signal: options.signal });
    },
  };
}

class RuntimeImpl implements Runtime {
  readonly control: RuntimeControl;
  readonly session: SessionStore;
  readonly resources: ResourceCatalog;
  readonly extensions: ExtensionHost | undefined;

  private activePath: SessionEntry[];
  private persistedMessageCount: number;
  private activeLeafId: string | null;
  private operationTail: Promise<void> = Promise.resolve();
  private poisoned = false;
  private poisonCause: unknown;
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(
    private readonly agent: Agent,
    initialPath: readonly SessionEntry[],
    private readonly deps: RuntimeDeps,
  ) {
    this.activePath = [...structuredClone(initialPath)];
    this.persistedMessageCount = messagesIn(initialPath).length;
    this.activeLeafId = initialPath.at(-1)?.id ?? null;
    this.session = deps.session;
    this.resources = deps.resources;
    this.extensions = deps.extensionHost;
    this.control = {
      getState: () => this.agent.getState(),
      subscribe: (listener) => this.agent.subscribe(listener),
      steer: (value) => this.agent.steer(value),
      followUp: (value) => this.agent.followUp(value),
      abort: () => this.agent.abort(),
    };
  }

  getActiveLeafId(): string | null {
    return this.activeLeafId;
  }

  private assertHealthy(): void {
    if (this.poisoned) throw this.poisonCause;
  }

  private async persist(
    messages: readonly AgentMessage[],
  ): Promise<void> {
    let parentId = this.activeLeafId;
    for (const message of messages) {
      try {
        const entry: MessageSessionEntry = {
          id: this.deps.createId(),
          parentId,
          timestamp: this.deps.now(),
          type: "message",
          message: structuredClone(message),
        };
        await this.session.append(entry);
        this.activePath.push(structuredClone(entry));
        this.persistedMessageCount += 1;
        this.activeLeafId = entry.id;
        parentId = entry.id;
      } catch (error) {
        this.poisoned = true;
        this.poisonCause = error;
        throw error;
      }
    }
  }

  prompt(value: string): Promise<AgentRunResult> {
    if (this.disposed) {
      return Promise.reject(new Error("Runtime 已 dispose"));
    }
    if (this.poisoned) {
      return Promise.reject(this.poisonCause);
    }

    const operation = this.operationTail.then(async () => {
      // 是否接受 prompt 只在调用时决定。dispose 可以关闭后续入口，
      // 但不能取消此前已经排入队列的工作。
      this.assertHealthy();
      const beforeCount = this.agent.getState().messages.length;
      const result = await this.agent.prompt(value);
      const suffix = result.messages.slice(beforeCount);
      await this.persist(suffix);
      return structuredClone(result);
    });
    this.operationTail = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  async flush(): Promise<void> {
    await this.operationTail;
    if (this.poisoned) throw this.poisonCause;
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flush();
    return this.disposePromise;
  }

  contextSnapshot(): ContextProjectionSnapshot {
    return {
      activePath: structuredClone(this.activePath),
      persistedMessageCount: this.persistedMessageCount,
    };
  }
}

function validateSessionSelection(
  entries: readonly SessionEntry[],
  activeLeafId: string | null,
): SessionEntry[] {
  if (entries.length === 0) {
    if (activeLeafId !== null) {
      throw new Error(
        "空 session 的 activeLeafId 必须是 null",
      );
    }
    return [];
  }
  if (activeLeafId === null) {
    throw new Error(
      "非空 session 必须显式提供 activeLeafId",
    );
  }
  return pathTo(entries, activeLeafId);
}

/**
 * composition root 只在这里把 session、context、resources、extensions、
 * tools 和有状态 Agent 接成一个对象图。
 */
export async function createRuntime(
  config: RuntimeConfig,
  deps: RuntimeDeps,
): Promise<Runtime> {
  const entries = await deps.session.entries();
  const initialPath = validateSessionSelection(
    entries,
    config.activeLeafId,
  );
  const initialMessages = messagesIn(initialPath);
  const systemPrompt = combinedSystemPrompt(
    config.systemPrompt,
    deps.resources,
  );

  let runtime!: RuntimeImpl;
  const model = createContextProjectingModel(
    deps.model,
    { systemPrompt, context: config.context },
    () => runtime.contextSnapshot(),
  );
  const coreExecutor: ToolExecutor = (call, context) =>
    executeToolCall(call, deps.tools, context);
  const toolExecutor = deps.extensionHost
    ? deps.extensionHost.wrapExecutor(coreExecutor)
    : coreExecutor;
  const agent = new Agent({
    model,
    tools: deps.tools,
    toolExecutor,
    initialMessages,
    systemPrompt,
    maxSteps: config.maxSteps,
  });
  runtime = new RuntimeImpl(agent, initialPath, deps);
  return runtime;
}

function finalAssistantText(result: AgentRunResult): string {
  const assistant = [...result.messages]
    .reverse()
    .find((message) => message.role === "assistant");
  return assistant ? textOf(assistant) : "";
}

/**
 * mode 不拥有 Agent 或 persistence，只负责调用 Runtime.prompt 并选择输出编码。
 */
export async function runMode(
  runtime: Runtime,
  mode: RuntimeMode,
  prompt: string,
  io: ModeIO,
): Promise<AgentRunResult> {
  if (
    mode !== "interactive" &&
    mode !== "print" &&
    mode !== "json"
  ) {
    throw new Error(`未知 runtime mode：${String(mode)}`);
  }
  const result = await runtime.prompt(prompt);
  const output =
    mode === "json"
      ? `${JSON.stringify({
          reason: result.reason,
          steps: result.steps,
          messages: result.messages,
        })}\n`
      : finalAssistantText(result);
  await io.write(output);
  return structuredClone(result);
}
