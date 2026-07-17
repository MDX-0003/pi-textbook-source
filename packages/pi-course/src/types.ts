export interface TextContent {
  type: "text";
  text: string;
}

export interface ToolCall {
  type: "toolCall";
  id: string;
  name: string;
  arguments: unknown;
  rawArguments?: string;
}

export type AssistantContent = TextContent | ToolCall;

export interface Usage {
  input: number;
  output: number;
  totalTokens: number;
}

export type StopReason =
  | "stop"
  | "length"
  | "toolUse"
  | "error"
  | "aborted";

export interface UserMessage {
  role: "user";
  content: TextContent[];
  timestamp: number;
}

export interface AssistantMessage {
  role: "assistant";
  content: AssistantContent[];
  provider: string;
  model: string;
  usage: Usage;
  stopReason: StopReason;
  errorMessage?: string;
  timestamp: number;
}

export interface ToolResultMessage<TDetails = unknown> {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: TextContent[];
  details?: TDetails;
  isError: boolean;
  timestamp: number;
}

export type AgentMessage =
  | UserMessage
  | AssistantMessage
  | ToolResultMessage;

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface AgentContext {
  systemPrompt?: string;
  messages: AgentMessage[];
  tools?: ToolDefinition[];
}

export type ModelEvent =
  | { type: "start"; partial: AssistantMessage }
  | {
      type: "text_delta";
      contentIndex: number;
      delta: string;
      partial: AssistantMessage;
    }
  | {
      type: "toolcall_delta";
      contentIndex: number;
      delta: string;
      partial: AssistantMessage;
    }
  | {
      type: "toolcall_end";
      contentIndex: number;
      toolCall: ToolCall;
      partial: AssistantMessage;
    }
  | {
      type: "done";
      reason: Extract<StopReason, "stop" | "length" | "toolUse">;
      message: AssistantMessage;
    }
  | {
      type: "error";
      reason: Extract<StopReason, "error" | "aborted">;
      error: AssistantMessage;
    };

export interface ModelStream extends AsyncIterable<ModelEvent> {
  result(): Promise<AssistantMessage>;
}

export interface Model {
  stream(
    context: AgentContext,
    options?: { signal?: AbortSignal },
  ): ModelStream;
}

export const EMPTY_USAGE: Usage = {
  input: 0,
  output: 0,
  totalTokens: 0,
};

export function text(value: string): TextContent {
  return { type: "text", text: value };
}

export function userMessage(value: string): UserMessage {
  return {
    role: "user",
    content: [text(value)],
    timestamp: Date.now(),
  };
}

export function textOf(message: AgentMessage): string {
  const blocks: readonly AssistantContent[] = message.content;
  return blocks.flatMap((block) =>
    block.type === "text" ? [block.text] : []
  ).join("\n");
}

export function assistantMessage(
  content: AssistantContent[],
  stopReason: StopReason = "stop",
  overrides: Partial<AssistantMessage> = {},
): AssistantMessage {
  return {
    role: "assistant",
    content,
    provider: "scripted",
    model: "scripted-v1",
    usage: EMPTY_USAGE,
    stopReason,
    timestamp: Date.now(),
    ...overrides,
  };
}
