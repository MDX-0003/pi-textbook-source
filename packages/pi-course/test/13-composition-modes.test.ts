import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import type { AgentRunResult } from "../src/agent-loop.js";
import {
  createContextProjectingModel,
  createRuntime,
  runMode,
  type ContextProjectionSnapshot,
  type Runtime,
  type RuntimeConfig,
  type RuntimeDeps,
  type RuntimeMode,
} from "../src/composition.js";
import {
  createExtensionHost,
  discoverResources,
  loadExtension,
  type ExtensionContext,
  type ResourceCatalog,
} from "../src/resources.js";
import { ScriptedModel } from "../src/scripted-model.js";
import type {
  MessageSessionEntry,
  SessionEntry,
  SessionStore,
} from "../src/session.js";
import {
  objectSchema,
  ToolRegistry,
  type Tool,
} from "../src/tool.js";
import {
  assistantMessage,
  text,
  textOf,
  userMessage,
  type AgentContext,
  type AgentMessage,
  type Model,
  type ToolCall,
} from "../src/types.js";

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error("等待可观察状态超时");
}

function emptyResources(): ResourceCatalog {
  return {
    resources: [],
    instructions: [],
    skills: [],
    templates: [],
  };
}

function config(
  activeLeafId: string | null,
  overrides: Partial<RuntimeConfig> = {},
): RuntimeConfig {
  return {
    activeLeafId,
    systemPrompt: "BASE SYSTEM",
    maxSteps: 8,
    context: {
      total: 1_000,
      reservedOutput: 0,
      safetyMargin: 0,
      estimateTokens: () => 1,
    },
    ...overrides,
  };
}

function messageEntry(
  id: string,
  parentId: string | null,
  message: AgentMessage,
): MessageSessionEntry {
  return {
    id,
    parentId,
    timestamp: 1,
    type: "message",
    message: structuredClone(message),
  };
}

class RecordingSessionStore implements SessionStore {
  readonly appendCalls: SessionEntry[] = [];
  readonly committed: SessionEntry[];

  constructor(
    initial: readonly SessionEntry[] = [],
    private readonly beforeCommit?: (
      entry: SessionEntry,
      index: number,
    ) => void | Promise<void>,
  ) {
    this.committed = [...structuredClone(initial)];
  }

  async append(entry: SessionEntry): Promise<void> {
    const snapshot = structuredClone(entry);
    const index = this.appendCalls.length;
    this.appendCalls.push(snapshot);
    await this.beforeCommit?.(structuredClone(snapshot), index);
    this.committed.push(snapshot);
  }

  async entries(): Promise<SessionEntry[]> {
    return structuredClone(this.committed);
  }
}

function runtimeDeps(
  model: Model,
  session: SessionStore,
  overrides: Partial<RuntimeDeps> = {},
): RuntimeDeps {
  let id = 0;
  let now = 100;
  return {
    model,
    tools: new ToolRegistry(),
    session,
    resources: emptyResources(),
    createId: () => `entry-${++id}`,
    now: () => ++now,
    ...overrides,
  };
}

function call(id: string, name = "probe"): ToolCall {
  return {
    type: "toolCall",
    id,
    name,
    arguments: {},
  };
}

function projectionSnapshot(
  activePath: readonly SessionEntry[],
): ContextProjectionSnapshot {
  return {
    activePath,
    persistedMessageCount: activePath.filter(
      (entry) => entry.type === "message",
    ).length,
  };
}

test("Lab 13.1 · 空 session 创建唯一 Runtime 外壳，不从 control 暴露 prompt", async () => {
  const tools = new ToolRegistry();
  const extensions = createExtensionHost(tools, {
    hookTimeoutMs: 20,
  });
  const session = new RecordingSessionStore();
  const resources = emptyResources();
  const runtime = await createRuntime(
    config(null),
    runtimeDeps(new ScriptedModel([]), session, {
      tools,
      resources,
      extensionHost: extensions,
    }),
  );

  assert.equal(runtime.session, session);
  assert.equal(runtime.resources, resources);
  assert.equal(runtime.extensions, extensions);
  assert.equal(runtime.getActiveLeafId(), null);
  assert.equal("prompt" in runtime.control, false);
  assert.deepEqual(runtime.control.getState().messages, []);
});

test("Lab 13.1 · 非空 session 必须显式选择 leaf，并把所选 path 深复制给 Agent", async () => {
  const root = messageEntry("root", null, userMessage("root question"));
  const left = messageEntry(
    "left",
    "root",
    assistantMessage([text("left answer")]),
  );
  const right = messageEntry(
    "right",
    "root",
    assistantMessage([text("right answer")]),
  );
  const leakyEntries: SessionEntry[] = [root, left, right];
  const leakyStore: SessionStore = {
    async append() {},
    async entries() {
      return leakyEntries;
    },
  };

  const selected = await createRuntime(
    config("right", { systemPrompt: undefined }),
    runtimeDeps(new ScriptedModel([]), leakyStore),
  );
  assert.deepEqual(
    selected.control.getState().messages.map(textOf),
    ["root question", "right answer"],
  );
  assert.equal("prompt" in selected.control, false);
  const rightText = right.message.content[0];
  if (rightText?.type === "text") rightText.text = "mutated source";
  assert.deepEqual(
    selected.control.getState().messages.map(textOf),
    ["root question", "right answer"],
  );
  await assert.rejects(
    createRuntime(
      config(null),
      runtimeDeps(new ScriptedModel([]), leakyStore),
    ),
    /非空 session 必须显式提供 activeLeafId/,
  );
  await assert.rejects(
    createRuntime(
      config("missing"),
      runtimeDeps(
        new ScriptedModel([]),
        new RecordingSessionStore(),
      ),
    ),
    /空 session 的 activeLeafId 必须是 null/,
  );
});

test("Lab 13.2 · context adapter 把未持久化 suffix 临时接到 active path", async () => {
  const oldUser = userMessage("old question");
  const oldAssistant = assistantMessage([text("old answer")]);
  const activePath = [
    messageEntry("old-user", null, oldUser),
    messageEntry("old-assistant", "old-user", oldAssistant),
  ];
  const inner = new ScriptedModel([
    assistantMessage([text("projected")]),
  ]);
  const projected = createContextProjectingModel(
    inner,
    {
      systemPrompt: "SYSTEM",
      context: config(null).context,
    },
    () => projectionSnapshot(activePath),
  );
  const incoming: AgentContext = {
    systemPrompt: "UNPROJECTED",
    messages: [oldUser, oldAssistant, userMessage("temporary user")],
    tools: [],
  };

  const stream = projected.stream(incoming);
  const incomingText = incoming.messages[2]?.content[0];
  if (incomingText?.type === "text") incomingText.text = "mutated later";
  await stream.result();

  assert.equal(inner.requests[0]?.systemPrompt, "SYSTEM");
  assert.deepEqual(
    inner.requests[0]?.messages.map(textOf),
    ["old question", "old answer", "temporary user"],
  );
  assert.equal(textOf(incoming.messages[2]!), "mutated later");
});

test("Lab 13.2 · context adapter 先扣固定成本，再只保留完整的最近 interaction", async () => {
  const messages = [
    userMessage("question one"),
    assistantMessage([text("answer one")]),
    userMessage("question two"),
    assistantMessage([text("answer two")]),
  ];
  const activePath = [
    messageEntry("u1", null, messages[0]!),
    messageEntry("a1", "u1", messages[1]!),
    messageEntry("u2", "a1", messages[2]!),
    messageEntry("a2", "u2", messages[3]!),
  ];
  const inner = new ScriptedModel([
    assistantMessage([text("trimmed")]),
  ]);
  const projected = createContextProjectingModel(
    inner,
    {
      systemPrompt: "SYSTEM",
      context: {
        total: 4,
        reservedOutput: 1,
        safetyMargin: 1,
        estimateTokens: () => 1,
      },
    },
    () => projectionSnapshot(activePath),
  );

  await projected.stream({
    messages: [...messages, userMessage("latest question")],
  }).result();
  assert.deepEqual(
    inner.requests[0]?.messages.map(textOf),
    ["latest question"],
  );
  assert.equal(inner.requests[0]?.systemPrompt, "SYSTEM");
});

test("Lab 13.2 · context adapter 保留 tools 与 signal，同时隔离 inner model 的输入", async () => {
  class SignalModel implements Model {
    readonly requests: AgentContext[] = [];
    readonly signals: Array<AbortSignal | undefined> = [];
    private readonly scripted = new ScriptedModel([
      assistantMessage([text("done")]),
    ]);

    stream(
      context: AgentContext,
      options: { signal?: AbortSignal } = {},
    ) {
      this.requests.push(structuredClone(context));
      this.signals.push(options.signal);
      return this.scripted.stream(context, options);
    }
  }

  const inner = new SignalModel();
  const projected = createContextProjectingModel(
    inner,
    {
      context: config(null, { systemPrompt: undefined }).context,
    },
    () => ({
      activePath: [],
      persistedMessageCount: 0,
    }),
  );
  const controller = new AbortController();
  const incoming: AgentContext = {
    messages: [userMessage("temporary")],
    tools: [
      {
        name: "probe",
        description: "probe",
        parameters: { type: "object" },
      },
    ],
  };

  await projected.stream(incoming, {
    signal: controller.signal,
  }).result();
  assert.equal(inner.signals[0], controller.signal);
  assert.deepEqual(inner.requests[0]?.tools, incoming.tools);
  assert.notEqual(inner.requests[0]?.tools, incoming.tools);
  const parameters = incoming.tools?.[0]?.parameters;
  if (parameters) parameters.type = "mutated";
  assert.deepEqual(
    inner.requests[0]?.tools?.[0]?.parameters,
    { type: "object" },
  );
});

test("Lab 13.3 · Runtime 接通 resources 与 extension，并等持久化后串行 resolve", async (t) => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "pi-composition-resources-"),
  );
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(path.join(root, "AGENTS.md"), "PROJECT RULE", "utf8");
  const skillRoot = path.join(root, "skills", "review");
  await mkdir(skillRoot, { recursive: true });
  await writeFile(
    path.join(skillRoot, "SKILL.md"),
    [
      "---",
      "name: review",
      "description: Review code carefully",
      "---",
      "INACTIVE SKILL SECRET",
    ].join("\n"),
    "utf8",
  );
  const resources = await discoverResources([root]);

  let coreExecutions = 0;
  const probe: Tool<Record<string, never>> = {
    name: "probe",
    description: "probe core execution",
    schema: objectSchema({}),
    async execute() {
      coreExecutions += 1;
      return { content: [text("core ran")] };
    },
  };
  const tools = new ToolRegistry();
  tools.register(probe);
  const extensions = createExtensionHost(tools, {
    hookTimeoutMs: 20,
  });
  await loadExtension(
    { id: "deny-probe", path: "/virtual/deny-probe.js" },
    {
      isTrusted: () => true,
      importModule: async () => ({
        default(context: ExtensionContext) {
          context.on("beforeToolCall", () => ({
            decision: "deny",
            reason: "course policy",
          }));
        },
      }),
      host: extensions,
    },
  );

  const firstAppend = deferred<void>();
  const store = new RecordingSessionStore(
    [],
    async (_entry, index) => {
      if (index === 0) await firstAppend.promise;
    },
  );
  const model = new ScriptedModel([
    assistantMessage([call("call-1")], "toolUse"),
    assistantMessage([text("first answer")]),
    assistantMessage([text("second answer")]),
  ]);
  const runtime = await createRuntime(
    config(null),
    runtimeDeps(model, store, {
      tools,
      resources,
      extensionHost: extensions,
    }),
  );

  let firstSettled = false;
  const first = runtime.prompt("run probe").finally(() => {
    firstSettled = true;
  });
  await waitUntil(() => store.appendCalls.length === 1);
  const second = runtime.prompt("second");
  await Promise.resolve();
  assert.equal(firstSettled, false);
  assert.equal(model.requests.length, 2);

  firstAppend.resolve(undefined);
  const firstResult = await first;
  await second;
  assert.equal(coreExecutions, 0);
  const denied = firstResult.messages.find(
    (message) => message.role === "toolResult",
  );
  assert.ok(denied);
  assert.equal(denied.isError, true);
  assert.match(textOf(denied), /course policy/);

  const systemPrompt = model.requests[0]?.systemPrompt ?? "";
  assert.match(systemPrompt, /BASE SYSTEM/);
  const configuredIndex = systemPrompt.indexOf("BASE SYSTEM");
  const resourcesIndex = systemPrompt.indexOf("# Pi resources");
  assert.ok(configuredIndex >= 0);
  assert.ok(resourcesIndex > configuredIndex);
  assert.match(systemPrompt, /PROJECT RULE/);
  assert.match(systemPrompt, /review: Review code carefully/);
  assert.doesNotMatch(systemPrompt, /INACTIVE SKILL SECRET/);

  assert.deepEqual(
    store.appendCalls.map((entry) =>
      entry.type === "message" ? entry.message.role : entry.type
    ),
    [
      "user",
      "assistant",
      "toolResult",
      "assistant",
      "user",
      "assistant",
    ],
  );
  assert.deepEqual(
    store.appendCalls.map((entry) => [
      entry.id,
      entry.parentId,
    ]),
    [
      ["entry-1", null],
      ["entry-2", "entry-1"],
      ["entry-3", "entry-2"],
      ["entry-4", "entry-3"],
      ["entry-5", "entry-4"],
      ["entry-6", "entry-5"],
    ],
  );
  assert.equal(runtime.getActiveLeafId(), "entry-6");

  const emptyModel = new ScriptedModel([
    assistantMessage([text("no system prompt")]),
  ]);
  const emptyRuntime = await createRuntime(
    config(null, { systemPrompt: undefined }),
    runtimeDeps(emptyModel, new RecordingSessionStore()),
  );
  await emptyRuntime.prompt("empty resources");
  assert.equal(emptyModel.requests[0]?.systemPrompt, undefined);
});

test("Lab 13.3 · 只追加恢复历史之后的新 suffix，写入失败后 Runtime poison", async () => {
  const initial = [
    messageEntry("old-user", null, userMessage("old question")),
    messageEntry(
      "old-assistant",
      "old-user",
      assistantMessage([text("old answer")]),
    ),
  ];
  const diskError = new Error("disk append failed");
  const store = new RecordingSessionStore(
    initial,
    (_entry, index) => {
      if (index === 1) throw diskError;
    },
  );
  const model = new ScriptedModel([
    assistantMessage([text("new answer")]),
    assistantMessage([text("must not run")]),
  ]);
  const runtime = await createRuntime(
    config("old-assistant"),
    runtimeDeps(model, store),
  );

  await assert.rejects(runtime.prompt("new question"), diskError);
  assert.deepEqual(
    store.appendCalls.map((entry) =>
      entry.type === "message" ? textOf(entry.message) : entry.type
    ),
    ["new question", "new answer"],
  );
  assert.equal(store.committed.length, 3);
  assert.equal(runtime.getActiveLeafId(), "entry-1");
  await assert.rejects(runtime.prompt("again"), diskError);
  await assert.rejects(runtime.flush(), diskError);
  assert.equal(model.requests.length, 1);
  assert.equal(store.appendCalls.length, 2);
});

test("Lab 13.3 · flush 等待当前队列，dispose 等待落盘并永久关闭 prompt", async () => {
  const gate = deferred<void>();
  const store = new RecordingSessionStore(
    [],
    async (_entry, index) => {
      if (index === 0) await gate.promise;
    },
  );
  const runtime = await createRuntime(
    config(null),
    runtimeDeps(
      new ScriptedModel([
        assistantMessage([text("persist me")]),
        assistantMessage([text("persist queued prompt")]),
      ]),
      store,
    ),
  );

  const prompting = runtime.prompt("save");
  await waitUntil(() => store.appendCalls.length === 1);
  const acceptedBeforeDispose = runtime.prompt("already accepted");
  let flushed = false;
  const flushing = runtime.flush().then(() => {
    flushed = true;
  });
  const disposing = runtime.dispose();
  assert.equal(runtime.dispose(), disposing);
  await assert.rejects(runtime.prompt("too late"), /已 dispose/);
  await Promise.resolve();
  assert.equal(flushed, false);

  gate.resolve(undefined);
  await prompting;
  await acceptedBeforeDispose;
  await flushing;
  await disposing;
  await runtime.flush();
  assert.equal(flushed, true);
  assert.equal(store.committed.length, 4);
  assert.equal(runtime.getActiveLeafId(), "entry-4");
});

function modeRuntime(result: AgentRunResult): {
  runtime: Runtime;
  prompts: string[];
} {
  const prompts: string[] = [];
  const session = new RecordingSessionStore();
  const runtime: Runtime = {
    control: {
      getState: () => ({
        status: "idle",
        messages: [],
        streamingText: "",
        pendingToolCallIds: [],
        diagnostics: [],
      }),
      subscribe: () => () => {},
      steer: () => {},
      followUp: () => {},
      abort: () => {},
    },
    session,
    resources: emptyResources(),
    extensions: undefined,
    getActiveLeafId: () => null,
    async prompt(value) {
      prompts.push(value);
      return structuredClone(result);
    },
    async flush() {},
    async dispose() {},
  };
  return { runtime, prompts };
}

test("Lab 13.4 · interactive 与 print 都只调用一次 Runtime.prompt，并写最终文本", async () => {
  const result: AgentRunResult = {
    reason: "stop",
    steps: 1,
    messages: [
      userMessage("question"),
      assistantMessage([text("final text")], "stop", {
        timestamp: 2,
      }),
    ],
  };
  const { runtime, prompts } = modeRuntime(result);
  const output: string[] = [];
  const io = {
    write(value: string) {
      output.push(value);
    },
  };

  await runMode(runtime, "interactive", "same prompt", io);
  await runMode(runtime, "print", "same prompt", io);
  assert.deepEqual(prompts, ["same prompt", "same prompt"]);
  assert.deepEqual(output, ["final text", "final text"]);
});

test("Lab 13.4 · json 只改变呈现；未知 mode 先失败，IO 错误原样上抛", async () => {
  const result: AgentRunResult = {
    reason: "length",
    steps: 2,
    messages: [
      userMessage("question"),
      assistantMessage([text("partial")], "length", {
        timestamp: 2,
      }),
    ],
  };
  const { runtime, prompts } = modeRuntime(result);
  const output: string[] = [];
  await runMode(runtime, "json", "json prompt", {
    write(value) {
      output.push(value);
    },
  });
  assert.equal(
    output[0],
    `${JSON.stringify({
      reason: result.reason,
      steps: result.steps,
      messages: result.messages,
    })}\n`,
  );
  assert.deepEqual(prompts, ["json prompt"]);

  await assert.rejects(
    runMode(
      runtime,
      "rpc" as RuntimeMode,
      "must not run",
      { write() {} },
    ),
    /未知 runtime mode/,
  );
  assert.deepEqual(prompts, ["json prompt"]);

  const sinkError = new Error("output sink failed");
  await assert.rejects(
    runMode(runtime, "print", "written fact", {
      write() {
        throw sinkError;
      },
    }),
    sinkError,
  );
  assert.deepEqual(prompts, ["json prompt", "written fact"]);
});
