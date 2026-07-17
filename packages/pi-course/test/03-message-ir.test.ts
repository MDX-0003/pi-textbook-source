import assert from "node:assert/strict";
import test from "node:test";
import { AssistantMessageEventStream } from "../src/event-stream.js";
import {
  assistantMessage,
  text,
  textOf,
} from "../src/types.js";

test("文本投影有损，但不修改 canonical content", () => {
  const message = assistantMessage([
    text("先读取"),
    {
      type: "toolCall",
      id: "c1",
      name: "read",
      arguments: { path: "README.md" },
    },
    text("再回答"),
  ]);
  const before = structuredClone(message.content);
  assert.equal(textOf(message), "先读取\n再回答");
  assert.deepEqual(message.content, before);
});

test(
  "error 也是协议终态，result resolve 最终消息",
  { timeout: 1_000 },
  async () => {
    const stream = new AssistantMessageEventStream();
    const error = assistantMessage([text("partial")], "error", {
      errorMessage: "socket reset",
    });
    stream.push({ type: "error", reason: "error", error });

    const observed = [];
    for await (const event of stream) observed.push(event.type);
    assert.deepEqual(observed, ["error"]);
    const result = await stream.result();
    assert.strictEqual(result, error);
    assert.equal(result.errorMessage, "socket reset");
  },
);
