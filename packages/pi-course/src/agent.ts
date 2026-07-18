import {
  runAgentLoop,
  type AgentRunResult,
  type LoopEvent,
} from "./agent-loop.js";
import type { ToolExecutor, ToolRegistry } from "./tool.js";
import {
  assistantMessage,
  userMessage,
  type AgentMessage,
  type Model,
  type UserMessage,
} from "./types.js";

export interface AgentOptions {
  model: Model;
  tools: ToolRegistry;
  toolExecutor?: ToolExecutor;
  initialMessages?: readonly AgentMessage[];
  systemPrompt?: string;
  maxSteps?: number;
}

export interface AgentState {
  status: "idle" | "running";
  messages: AgentMessage[];
  activeRunId?: number;
  lastReason?: AgentRunResult["reason"];
  streamingText: string;
  pendingToolCallIds: string[];
  diagnostics: string[];
}

export type AgentEvent =
  | { type: "run_start"; runId: number; message: UserMessage }
  | { type: "loop"; runId: number; event: LoopEvent }
  | { type: "run_end"; runId: number; result: AgentRunResult };

function clone<T>(value: T): T {
  return structuredClone(value);
}

export function reduceAgentState(
  state: AgentState,
  event: AgentEvent,
): AgentState {
  if (event.type === "run_start") {
    return {
      ...state,
      status: "running",
      activeRunId: event.runId,
      lastReason: undefined,
      messages: [...clone(state.messages), clone(event.message)],
      streamingText: "",
      pendingToolCallIds: [],
    };
  }

  if (
    state.status !== "running" ||
    state.activeRunId !== event.runId
  ) {
    return state;
  }

  if (event.type === "loop") {
    const loop = event.event;
    if (
      loop.type === "model_event" &&
      loop.event.type === "text_delta"
    ) {
      return {
        ...state,
        streamingText: state.streamingText + loop.event.delta,
      };
    }
    if (loop.type === "tool_start") {
      return {
        ...state,
        pendingToolCallIds: [
          ...state.pendingToolCallIds,
          loop.call.id,
        ],
      };
    }
    if (loop.type === "tool_end" || loop.type === "tool_skipped") {
      return {
        ...state,
        pendingToolCallIds: state.pendingToolCallIds.filter(
          (id) => id !== loop.result.toolCallId,
        ),
      };
    }
    if (loop.type === "assistant_message") {
      return { ...state, streamingText: "" };
    }
    return state;
  }

  return {
    status: "idle",
    messages: clone(event.result.messages),
    lastReason: event.result.reason,
    streamingText: "",
    pendingToolCallIds: [],
    diagnostics: [...state.diagnostics],
  };
}

interface ActiveRun {
  id: number;
  controller: AbortController;
  steering: UserMessage[];
  followUps: UserMessage[];
  acceptingInput: boolean;
}

export class Agent {
  private state: AgentState;
  private readonly subscribers = new Set<(event: AgentEvent) => void>();
  private readonly pendingEvents: AgentEvent[] = [];
  private dispatchingEvents = false;
  private activeRun?: ActiveRun;
  private nextRunId = 1;

  constructor(private readonly options: AgentOptions) {
    this.state = {
      status: "idle",
      messages: [...clone(options.initialMessages ?? [])],
      streamingText: "",
      pendingToolCallIds: [],
      diagnostics: [],
    };
  }

  getState(): AgentState {
    return clone(this.state);
  }

  subscribe(listener: (event: AgentEvent) => void): () => void {
    this.subscribers.add(listener);
    return () => this.subscribers.delete(listener);
  }

  private emit(event: AgentEvent): void {
    this.pendingEvents.push(event);
    if (this.dispatchingEvents) return;

    this.dispatchingEvents = true;
    try {
      while (this.pendingEvents.length > 0) {
        const next = this.pendingEvents.shift()!;
        this.state = reduceAgentState(this.state, next);
        for (const listener of [...this.subscribers]) {
          try {
            listener(clone(next));
          } catch (error) {
            this.state = {
              ...this.state,
              diagnostics: [
                ...this.state.diagnostics,
                `subscriber: ${
                  error instanceof Error ? error.message : String(error)
                }`,
              ],
            };
          }
        }
      }
    } finally {
      this.dispatchingEvents = false;
    }
  }

  steer(value: string): void {
    if (!this.activeRun?.acceptingInput) {
      throw new Error("steering 只在当前 run 尚未结束时有意义");
    }
    this.activeRun.steering.push(userMessage(value));
  }

  followUp(value: string): void {
    if (!this.activeRun?.acceptingInput) {
      throw new Error("follow-up 只在当前 run 尚未结束时排队");
    }
    this.activeRun.followUps.push(userMessage(value));
  }

  abort(): void {
    this.activeRun?.controller.abort();
  }

  async prompt(value: string): Promise<AgentRunResult> {
    if (this.activeRun) {
      throw new Error("Agent is busy");
    }

    const message = userMessage(value);
    const contextMessages = [...clone(this.state.messages), clone(message)];
    const run: ActiveRun = {
      id: this.nextRunId++,
      controller: new AbortController(),
      steering: [],
      followUps: [],
      acceptingInput: true,
    };
    this.activeRun = run;
    this.emit({ type: "run_start", runId: run.id, message });

    let result: AgentRunResult;
    try {
      result = await runAgentLoop({
        model: this.options.model,
        tools: this.options.tools,
        context: {
          systemPrompt: this.options.systemPrompt,
          messages: contextMessages,
        },
        signal: run.controller.signal,
        maxSteps: this.options.maxSteps,
        executeToolCall: this.options.toolExecutor,
        takeSteeringMessages: () => run.steering.splice(0),
        takeFollowUpMessages: () => run.followUps.splice(0),
        onEvent: (event) => {
          if (event.type === "turn_end") run.acceptingInput = false;
          this.emit({ type: "loop", runId: run.id, event });
        },
      });
    } catch (error) {
      const failed = assistantMessage([], "error", {
        errorMessage:
          error instanceof Error ? error.message : String(error),
      });
      result = {
        reason: "error",
        messages: [...contextMessages, failed],
        steps: 0,
      };
    } finally {
      if (this.activeRun === run) this.activeRun = undefined;
      run.steering.splice(0);
      run.followUps.splice(0);
    }

    this.emit({ type: "run_end", runId: run.id, result });
    return clone(result);
  }
}
