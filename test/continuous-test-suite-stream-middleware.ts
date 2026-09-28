#!/usr/bin/env tsx
import "dotenv/config";

/**
 * Public generate()/stream() middleware contracts for the OpenAI-compatible
 * family. Local HTTP fixtures prove prompt rewrites, real tool execution,
 * guardrail blocking/filtering, error propagation, completion and cancellation.
 * Runtime imports use only the built entry; type-only imports are erased.
 *
 * Run: pnpm run build && pnpm run test:stream-middleware
 */

import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { once } from "node:events";
import { z } from "zod";
import type {
  NeuroLinkMiddleware,
  LanguageModelV3StreamPart,
  StreamResult,
  AnalyticsData,
  MiddlewareFactoryOptions,
  OnFinishCallback,
  OnErrorCallback,
  OnChunkCallback,
  LifecycleFinishPayload,
  LifecycleErrorPayload,
  LifecycleChunkPayload,
} from "../src/lib/types/index.js";
import { defineSuite } from "./helpers/harness.js";
import { assertDistFresh } from "./helpers/distFreshness.js";
import {
  mockOpenAICredentials,
  startMockChatServer,
  startScriptedChatServer,
  chatCompletion,
} from "./helpers/mockChatServer.js";
import { NeuroLink, tool, logger } from "../dist/index.js";

assertDistFresh();

const { test, section, runSuite } = defineSuite("Stream middleware", {
  offline: true,
});

const MARKER = "MIDDLEWARE_TOUCHED_THE_PROMPT";

type ProbeRecord = {
  transformParamsCalls: Array<"generate" | "stream">;
  wrapGenerateCalls: number;
  wrapStreamCalls: number;
};

/**
 * A middleware that records which hooks fired and rewrites the outgoing
 * prompt. The rewrite is what separates "the hook ran" from "the hook
 * affected the request" — the stand-in captures the wire body, so the
 * marker either reached the provider or it did not.
 *
 * `metadata` is mandatory. A probe without it registers under an `undefined`
 * id and silently never runs, which measures the probe rather than the code.
 */
const createProbe = (record: ProbeRecord): NeuroLinkMiddleware => ({
  specificationVersion: "v3",
  metadata: { id: "stream-probe", name: "Stream probe" },
  transformParams: async ({ type, params }) => {
    record.transformParamsCalls.push(type);
    return {
      ...params,
      prompt: [
        ...params.prompt,
        { role: "user", content: [{ type: "text", text: MARKER }] },
      ],
    };
  },
  wrapGenerate: async ({ doGenerate }) => {
    record.wrapGenerateCalls += 1;
    return doGenerate();
  },
  wrapStream: async ({ doStream }) => {
    record.wrapStreamCalls += 1;
    return doStream();
  },
});

const emptyRecord = (): ProbeRecord => ({
  transformParamsCalls: [],
  wrapGenerateCalls: 0,
  wrapStreamCalls: 0,
});

const middlewareOptions = (record: ProbeRecord) => ({
  middleware: [createProbe(record)],
  enabledMiddleware: ["stream-probe"],
});

// ---------------------------------------------------------------------------
// PRECONDITION — the probe is correctly registered and does run on generate.
// ---------------------------------------------------------------------------

await test("generate applies model middleware (precondition)", async () => {
  const server = await startMockChatServer();
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    await nl.generate({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      credentials: mockOpenAICredentials(server),
      middleware: middlewareOptions(record),
    });

    if (record.transformParamsCalls.length === 0) {
      throw new Error(
        "probe never ran on generate — the probe is mis-registered, so the " +
          "stream assertions below would be meaningless",
      );
    }
    if (record.wrapGenerateCalls === 0) {
      throw new Error("wrapGenerate never fired on the generate path");
    }
    const body = server.getLastRequestBody();
    if (!body || !body.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on generate",
      );
    }
  } finally {
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// THE GAP — the same middleware, the same provider, the streaming path.
// ---------------------------------------------------------------------------

await test("stream applies model middleware", async () => {
  const server = await startMockChatServer();
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    const result = await nl.stream({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      credentials: mockOpenAICredentials(server),
      middleware: middlewareOptions(record),
    });

    for await (const _chunk of result.stream) {
      // Drain. The assertions are about what was sent, not what came back.
    }

    if (!server.wasCalled()) {
      throw new Error(
        "the stand-in was never called — the stream never left the machine, " +
          "so nothing below can be concluded about middleware",
      );
    }
    if (record.transformParamsCalls.length === 0) {
      throw new Error("transformParams never fired on the streaming path");
    }
    if (!record.transformParamsCalls.includes("stream")) {
      throw new Error(
        'transformParams fired but never with type "stream" on stream()',
      );
    }
    if (record.wrapStreamCalls === 0) {
      throw new Error("wrapStream never fired on the streaming path");
    }
  } finally {
    await server.close();
  }
});

await test("a stream transformParams rewrite reaches the wire", async () => {
  const server = await startMockChatServer();
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    const result = await nl.stream({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      credentials: mockOpenAICredentials(server),
      middleware: middlewareOptions(record),
    });

    for await (const _chunk of result.stream) {
      // Drain.
    }

    const body = server.getLastRequestBody();
    if (!body) {
      throw new Error("the stand-in captured no request body for the stream");
    }
    if (!body.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on stream",
      );
    }
  } finally {
    await server.close();
  }
});

const readText = async (result: StreamResult): Promise<string> => {
  let text = "";
  for await (const chunk of result.stream) {
    if ("content" in chunk && typeof chunk.content === "string") {
      text += chunk.content;
    }
  }
  return text;
};

// The deadline is generous on purpose. The two cancellation cases below
// failed once with "request never reached the fixture" while a 53-suite sweep
// saturated the CPU in parallel — stream setup alone exceeded a 3s bound before
// the HTTP request went out — and passed 20/20 in isolation. A bound exists to
// catch a hang, not to race the scheduler.
const bounded = async <T>(
  promise: PromiseLike<T>,
  deadlineMs = 30_000,
): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve(promise),
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("middleware completion did not settle")),
          deadlineMs,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

await test("wrapStream filters text and observes the V3 terminal event", async () => {
  const server = await startMockChatServer();
  const sdk = new NeuroLink();
  const seen: LanguageModelV3StreamPart[] = [];
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "wire-filter", name: "Wire filter" },
    wrapStream: async ({ doStream }) => {
      const result = await doStream();
      return {
        ...result,
        stream: result.stream.pipeThrough(
          new TransformStream({
            transform(part: LanguageModelV3StreamPart, controller) {
              seen.push(part);
              controller.enqueue(
                part.type === "text-delta"
                  ? { ...part, delta: part.delta.toUpperCase() }
                  : part,
              );
            },
          }),
        ),
      };
    },
  };
  try {
    const result = await sdk.stream({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      disableInternalFallback: true,
      credentials: mockOpenAICredentials(server),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["wire-filter"],
      },
    });
    assert.equal(
      await bounded(readText(result)),
      "MOCK REPLY",
      "filtered text lost",
    );
    assert.ok(server.wasCalled(), "wire request did not run");
    assert.equal(
      seen.filter((part) => part.type === "finish").length,
      1,
      "terminal event not forwarded",
    );
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

await test("a synthetic blocking stream delivers text and settles analytics without HTTP", async () => {
  const server = await startMockChatServer();
  const sdk = new NeuroLink();
  let invoked = false;
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "block-stream", name: "Block stream" },
    wrapStream: async () => {
      invoked = true;
      return {
        stream: new ReadableStream<LanguageModelV3StreamPart>({
          start(controller) {
            controller.enqueue({ type: "text-start", id: "blocked" });
            controller.enqueue({
              type: "text-delta",
              id: "blocked",
              delta: "BLOCKED",
            });
            controller.enqueue({ type: "text-end", id: "blocked" });
            controller.enqueue({
              type: "finish",
              finishReason: { unified: "stop" },
              usage: { inputTokens: { total: 0 }, outputTokens: { total: 0 } },
            });
            controller.close();
          },
        }),
      };
    },
  };
  try {
    const result = await sdk.stream({
      input: { text: "block this" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      disableInternalFallback: true,
      enableAnalytics: true,
      credentials: mockOpenAICredentials(server),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["block-stream"],
      },
    });
    assert.equal(
      await bounded(readText(result)),
      "BLOCKED",
      "blocked content lost",
    );
    assert.ok(invoked, "blocking middleware did not run");
    assert.equal(
      server.getAllRequestBodies().length,
      0,
      "blocked request reached HTTP",
    );
    assert.ok(result.analytics, "analytics were not exposed");
    await bounded(Promise.resolve(result.analytics));
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

await test("a synthetic blocking stream with cache-inclusive usage bills the cached portion at the cache-read rate", async () => {
  // Same mechanism as "a synthetic blocking stream ... without HTTP" above —
  // a wrapStream that never calls the given doStream() is the only way the
  // OpenAI-compatible provider's executeStream() reaches its `!loopPromise`
  // usage branch — but this time with realistic OpenAI-style cache-inclusive
  // usage (`inputTokens.total` includes the cached portion) instead of all
  // zero, so the branch's cache split is actually exercised. gpt-4o-mini's
  // published rates (src/lib/models/manifests/openai.ts) are input
  // $0.15/MTok, output $0.60/MTok, cacheRead $0.0375/MTok: 200 uncached +
  // 800 cached + 100 output tokens price to 0.00003 + 0.00003 + 0.00006.
  const server = await startMockChatServer();
  const sdk = new NeuroLink();
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "cache-usage-stream", name: "Cache usage stream" },
    wrapStream: async () => ({
      stream: new ReadableStream<LanguageModelV3StreamPart>({
        start(controller) {
          controller.enqueue({ type: "text-start", id: "cached" });
          controller.enqueue({
            type: "text-delta",
            id: "cached",
            delta: "cached reply",
          });
          controller.enqueue({ type: "text-end", id: "cached" });
          controller.enqueue({
            type: "finish",
            finishReason: { unified: "stop" },
            usage: {
              inputTokens: { total: 1000, cacheRead: 800 },
              outputTokens: { total: 100 },
            },
          });
          controller.close();
        },
      }),
    }),
  };
  try {
    const result = await sdk.stream({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      disableInternalFallback: true,
      enableAnalytics: true,
      credentials: mockOpenAICredentials(server),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["cache-usage-stream"],
      },
    });
    await bounded(readText(result));
    assert.equal(
      server.getAllRequestBodies().length,
      0,
      "synthetic stream did not stay off the wire",
    );
    assert.ok(result.analytics, "analytics were not exposed");
    const analytics = await bounded(Promise.resolve(result.analytics));
    const { tokenUsage, cost } = analytics as AnalyticsData;
    assert.equal(
      tokenUsage.input,
      200,
      "uncached input was not separated out of the cache-inclusive total",
    );
    assert.equal(
      tokenUsage.cacheReadTokens,
      800,
      "cache-read tokens were not reported on the streaming no-tool-call path",
    );
    assert.equal(
      tokenUsage.total,
      1100,
      "the reported total drifted when the cache split was introduced",
    );
    assert.equal(
      cost,
      0.00012,
      "the cached portion was not billed at the discounted cache-read rate",
    );
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

for (const mode of ["generate", "stream"] as const) {
  await test(`guardrail bad-word filtering changes ${mode} content`, async () => {
    const server = await startMockChatServer();
    const sdk = new NeuroLink();
    try {
      const options = {
        input: { text: "hello" },
        provider: "openai",
        model: "gpt-4o-mini",
        disableTools: true,
        disableInternalFallback: true,
        credentials: mockOpenAICredentials(server),
        middleware: {
          middlewareConfig: {
            guardrails: {
              enabled: true,
              config: {
                badWords: {
                  enabled: true,
                  list: ["mock"],
                  replacementText: "CLEAN",
                },
              },
            },
          },
        },
      };
      const content =
        mode === "generate"
          ? (await sdk.generate(options)).content
          : await bounded(readText(await sdk.stream(options)));
      assert.ok(server.wasCalled(), "provider was not exercised");
      assert.equal(
        content,
        "CLEAN reply",
        "guardrail filtering did not reach consumer",
      );
    } finally {
      await sdk.shutdown();
      await server.close();
    }
  });

  await test(`precall guardrail blocks ${mode} after evaluator returns unsafe`, async () => {
    const evaluator = await startScriptedChatServer([
      chatCompletion({
        content: JSON.stringify({
          overall: "unsafe",
          safetyScore: 1,
          appropriatenessScore: 1,
          confidenceLevel: 10,
          suggestedAction: "block",
          reasoning: "Deterministic blocking fixture",
        }),
      }),
    ]);
    const target = await startMockChatServer();
    const saved = {
      key: process.env.OPENAI_COMPATIBLE_API_KEY,
      url: process.env.OPENAI_COMPATIBLE_BASE_URL,
    };
    process.env.OPENAI_COMPATIBLE_API_KEY = "test-evaluator-key";
    process.env.OPENAI_COMPATIBLE_BASE_URL = evaluator.baseURL;
    const sdk = new NeuroLink();
    try {
      const options = {
        input: { text: "block this request" },
        provider: "openai",
        model: "gpt-4o-mini",
        disableTools: true,
        disableInternalFallback: true,
        enableAnalytics: true,
        credentials: mockOpenAICredentials(target),
        middleware: {
          middlewareConfig: {
            guardrails: {
              enabled: true,
              config: {
                precallEvaluation: {
                  enabled: true,
                  provider: "openai-compatible",
                  evaluationModel: "fixture-evaluator",
                },
              },
            },
          },
        },
      };
      const result =
        mode === "generate"
          ? await sdk.generate(options)
          : await sdk.stream(options);
      const content =
        "stream" in result ? await bounded(readText(result)) : result.content;
      assert.ok(evaluator.wasCalled(), "guardrail evaluator was not exercised");
      assert.equal(
        target.getAllRequestBodies().length,
        0,
        "blocked input reached target provider",
      );
      assert.equal(
        content,
        "Request contains inappropriate content and has been blocked.",
        "guardrail refusal was lost",
      );
      if (result.analytics) {
        await bounded(Promise.resolve(result.analytics));
      }
    } finally {
      if (saved.key === undefined) {
        delete process.env.OPENAI_COMPATIBLE_API_KEY;
      } else {
        process.env.OPENAI_COMPATIBLE_API_KEY = saved.key;
      }
      if (saved.url === undefined) {
        delete process.env.OPENAI_COMPATIBLE_BASE_URL;
      } else {
        process.env.OPENAI_COMPATIBLE_BASE_URL = saved.url;
      }
      await sdk.shutdown();
      await target.close();
      await evaluator.close();
    }
  });

  await test(`${mode} executes a real tool round trip with middleware enabled`, async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const server = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) {
        chunks.push(Buffer.from(chunk));
      }
      const body: Record<string, unknown> = JSON.parse(
        Buffer.concat(chunks).toString(),
      );
      bodies.push(body);
      const first = bodies.length === 1;
      const call = {
        id: "call_fixture",
        type: "function",
        function: { name: "lookup", arguments: '{"value":7}' },
      };
      if (body.stream === true) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        const delta = first
          ? { tool_calls: [{ index: 0, ...call }] }
          : { content: "answer 42" };
        for (const data of [
          { choices: [{ index: 0, delta, finish_reason: null }] },
          {
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: first ? "tool_calls" : "stop",
              },
            ],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          },
        ]) {
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
        res.end("data: [DONE]\n\n");
      } else {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify(
            chatCompletion(
              first
                ? { finishReason: "tool_calls", toolCalls: [call] }
                : { content: "answer 42" },
            ),
          ),
        );
      }
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const sdk = new NeuroLink();
    let calls = 0;
    try {
      const options = {
        input: { text: "call lookup" },
        provider: "openai",
        model: "gpt-4o-mini",
        maxSteps: 3,
        disableInternalFallback: true,
        enabledToolNames: ["lookup"],
        credentials: {
          openai: {
            apiKey: "test-key",
            baseURL: `http://127.0.0.1:${address.port}/v1`,
          },
        },
        tools: {
          lookup: tool({
            inputSchema: z.object({ value: z.number() }),
            execute: async ({ value }) => {
              calls++;
              assert.equal(value, 7, "tool arguments changed");
              return { answer: 42 };
            },
          }),
        },
        middleware: middlewareOptions(emptyRecord()),
      };
      const result =
        mode === "generate"
          ? await sdk.generate(options)
          : await sdk.stream(options);
      const content =
        "stream" in result ? await bounded(readText(result)) : result.content;
      assert.equal(calls, 1, "tool was not invoked once");
      assert.equal(
        bodies.length,
        2,
        "tool round trip did not send two requests",
      );
      assert.ok(
        JSON.stringify(bodies[1].messages).includes("42"),
        "tool result not sent back",
      );
      assert.equal(content, "answer 42", "final tool answer lost");
    } finally {
      await sdk.shutdown();
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
}

await test("breaking out of a wrapped stream cancels the upstream socket", async () => {
  let closed = false;
  let received = false;
  const server = createServer(async (req, res) => {
    for await (const _chunk of req) {
      /* Read the request before streaming. */
    }
    received = true;
    res.on("close", () => {
      closed = true;
    });
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(
      `data: ${JSON.stringify({ choices: [{ delta: { content: "first" }, finish_reason: null }] })}\n\n`,
    );
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const sdk = new NeuroLink();
  try {
    const result = await sdk.stream({
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      disableInternalFallback: true,
      credentials: {
        openai: {
          apiKey: "test-key",
          baseURL: `http://127.0.0.1:${address.port}/v1`,
        },
      },
      middleware: middlewareOptions(emptyRecord()),
    });
    await bounded(
      (async () => {
        for await (const chunk of result.stream) {
          if ("content" in chunk && chunk.content) {
            break;
          }
        }
      })(),
    );
    assert.ok(received, "server was never reached");
    await bounded(
      (async () => {
        while (!closed) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      })(),
    );
    assert.ok(closed, "upstream socket not closed");
  } finally {
    await sdk.shutdown();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

for (const mode of ["generate", "stream"] as const) {
  await test(`${mode} propagates provider failure with middleware enabled`, async () => {
    const server = await startScriptedChatServer([
      { status: 400, body: { error: { message: "fixture rejected" } } },
    ]);
    const sdk = new NeuroLink();
    let caught: unknown;
    try {
      const options = {
        input: { text: "hello" },
        provider: "openai",
        model: "gpt-4o-mini",
        disableTools: true,
        disableInternalFallback: true,
        credentials: mockOpenAICredentials(server),
        middleware: middlewareOptions(emptyRecord()),
      };
      try {
        if (mode === "generate") {
          await sdk.generate(options);
        } else {
          await readText(await sdk.stream(options));
        }
      } catch (error) {
        caught = error;
      }
      assert.ok(server.wasCalled(), "failure fixture was not reached");
      assert.ok(caught, "provider failure was swallowed");
    } finally {
      await sdk.shutdown();
      await server.close();
    }
  });

  for (const end of ["abort", "timeout"] as const) {
    await test(`${mode} ${end} closes an in-flight HTTP request`, async () => {
      const controller = new AbortController();
      let received = false;
      let closed = false;
      const server = createServer(async (req, res) => {
        for await (const _part of req) {
          /* Ensure the model request arrived. */
        }
        received = true;
        res.on("close", () => {
          closed = true;
        });
        if (end === "abort") {
          controller.abort(new Error("fixture abort"));
        }
      });
      server.listen(0, "127.0.0.1");
      await once(server, "listening");
      const address = server.address();
      assert.ok(address && typeof address === "object");
      const sdk = new NeuroLink();
      let caught: unknown;
      try {
        const options = {
          input: { text: "hello" },
          provider: "openai",
          model: "gpt-4o-mini",
          disableTools: true,
          disableInternalFallback: true,
          abortSignal: controller.signal,
          timeout: 1000,
          turnTimeoutMs: 1500,
          credentials: {
            openai: {
              apiKey: "test-key",
              baseURL: `http://127.0.0.1:${address.port}/v1`,
            },
          },
          middleware: middlewareOptions(emptyRecord()),
        };
        try {
          await bounded(
            (async () => {
              if (mode === "generate") {
                await sdk.generate(options);
              } else {
                await readText(await sdk.stream(options));
              }
            })(),
          );
        } catch (error) {
          caught = error;
        }
        assert.ok(received, "request never reached the fixture");
        assert.ok(caught, "cancellation did not terminate the call");
        assert.notEqual(
          caught instanceof Error ? caught.message : "",
          "middleware completion did not settle",
          "only the test deadline ended the call",
        );
        await bounded(
          (async () => {
            while (!closed) {
              await new Promise((resolve) => setTimeout(resolve, 10));
            }
          })(),
        );
        assert.ok(closed, "upstream request stayed open");
      } finally {
        controller.abort();
        await sdk.shutdown();
        server.closeAllConnections();
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    });
  }
}

await test("stream middleware sampling edits reach the wire and preserve native fields", async () => {
  const server = await startMockChatServer();
  const sdk = new NeuroLink();
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "sampling", name: "Sampling" },
    transformParams: async ({ params }) => ({
      ...params,
      maxOutputTokens: 77,
      temperature: 0.25,
      topP: 0.9,
    }),
  };
  try {
    await readText(
      await sdk.stream({
        provider: "openai",
        model: "gpt-4o-mini",
        input: { text: "hello" },
        disableTools: true,
        disableInternalFallback: true,
        maxTokens: 128,
        temperature: 0.7,
        credentials: mockOpenAICredentials(server),
        middleware: {
          middleware: [middleware],
          enabledMiddleware: ["sampling"],
        },
      }),
    );
    assert.ok(server.wasCalled(), "sampling fixture not reached");
    const body: Record<string, unknown> = JSON.parse(
      server.getLastRequestBody() ?? "{}",
    );
    assert.equal(body.max_tokens, 77, "token override lost");
    assert.equal(body.temperature, 0.25, "temperature override lost");
    assert.equal(body.top_p, 0.9, "top-p override lost");
    assert.equal(body.stream, true, "stream mode changed");
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

await test("generate schema recovery still works with middleware enabled", async () => {
  const server = await startScriptedChatServer([
    chatCompletion({ content: '{"answer":42}' }),
  ]);
  const sdk = new NeuroLink();
  const schema = z.object({ answer: z.number() });
  try {
    const result = await sdk.generate({
      provider: "openai",
      model: "gpt-4o-mini",
      input: { text: "answer" },
      schema,
      disableTools: true,
      disableInternalFallback: true,
      credentials: mockOpenAICredentials(server),
      middleware: middlewareOptions(emptyRecord()),
    });
    assert.ok(server.wasCalled(), "schema fixture not reached");
    assert.equal(
      schema.parse(result.structuredData).answer,
      42,
      "structured data changed",
    );
    assert.deepEqual(
      JSON.parse(result.content),
      result.structuredData,
      "JSON text disagrees with object",
    );
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

/**
 * A prohibited term split across a boundary. The bad-word filter used to run
 * on each text part (generate) and each text-delta (stream) in isolation, so a
 * term that straddled two of them passed unchanged and was reassembled
 * downstream — `lifecycle.ts` concatenates adjacent text parts, and every
 * stream consumer concatenates deltas. Each case first proves the split reaches
 * the consumer whole when no guardrail is configured, so a pass cannot come
 * from the pieces never having been split, and only then asserts the guardrail
 * catches the reassembled term.
 */
const SPLIT_TERM = ["inappro", "priate"] as const;
const WHOLE_TERM = SPLIT_TERM.join("");

const splitTermGuardrails = {
  middlewareConfig: {
    guardrails: {
      enabled: true,
      config: {
        badWords: {
          enabled: true,
          list: [WHOLE_TERM],
          replacementText: "CLEAN",
        },
      },
    },
  },
};

type SplitServer = {
  baseURL: string;
  requests: () => number;
  close: () => Promise<void>;
};

const readRequestBody = (req: IncomingMessage): Promise<string> =>
  new Promise((resolve) => {
    let raw = "";
    req.setEncoding("utf8");
    req.on("data", (chunk: string) => {
      raw += chunk;
    });
    req.on("end", () => resolve(raw));
  });

const listen = async (
  server: Server,
): Promise<{ port: number; close: () => Promise<void> }> => {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  return {
    port: typeof address === "object" && address ? address.port : 0,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
};

/** OpenAI-compatible server that streams each piece as its own SSE chunk. */
const startSplitDeltaChatServer = async (
  pieces: ReadonlyArray<string>,
): Promise<SplitServer> => {
  let requests = 0;
  const chunk = (delta: Record<string, unknown>, finish: string | null) =>
    `data: ${JSON.stringify({
      id: "split",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  const server = createServer(async (req, res) => {
    await readRequestBody(req);
    requests += 1;
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    for (const piece of pieces) {
      res.write(chunk({ role: "assistant", content: piece }, null));
    }
    res.write(chunk({}, "stop"));
    res.write("data: [DONE]\n\n");
    res.end();
  });
  const { port, close } = await listen(server);
  return {
    baseURL: `http://127.0.0.1:${port}/v1`,
    requests: () => requests,
    close,
  };
};

/**
 * Anthropic Messages server answering with one text block per piece — which
 * the native Anthropic path turns into adjacent text parts in the V3 result.
 * Serves both the JSON and the SSE form so whichever the generate path uses,
 * the blocks stay split.
 */
const startSplitBlockAnthropicServer = async (
  pieces: ReadonlyArray<string>,
): Promise<SplitServer> => {
  let requests = 0;
  const model = "claude-sonnet-4-20250514";
  const server = createServer(async (req, res) => {
    const raw = await readRequestBody(req);
    requests += 1;
    const wantsStream = ((): boolean => {
      try {
        return (JSON.parse(raw) as { stream?: unknown }).stream === true;
      } catch {
        return false;
      }
    })();
    if (!wantsStream) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: "msg_split",
          type: "message",
          role: "assistant",
          model,
          content: pieces.map((text) => ({ type: "text", text })),
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 2 },
        }),
      );
      return;
    }
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const event = (type: string, data: Record<string, unknown>): void => {
      res.write(
        `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`,
      );
    };
    event("message_start", {
      message: {
        id: "msg_split",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        usage: { input_tokens: 1, output_tokens: 0 },
      },
    });
    pieces.forEach((text, index) => {
      event("content_block_start", {
        index,
        content_block: { type: "text", text: "" },
      });
      event("content_block_delta", {
        index,
        delta: { type: "text_delta", text },
      });
      event("content_block_stop", { index });
    });
    event("message_delta", {
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: 2 },
    });
    event("message_stop", {});
    res.end();
  });
  const { port, close } = await listen(server);
  return {
    baseURL: `http://127.0.0.1:${port}`,
    requests: () => requests,
    close,
  };
};

const withEnv = async <T>(
  vars: Record<string, string>,
  fn: () => Promise<T>,
): Promise<T> => {
  const saved = new Map(
    Object.keys(vars).map((key) => [key, process.env[key]] as const),
  );
  Object.assign(process.env, vars);
  try {
    return await fn();
  } finally {
    for (const [key, value] of saved) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }
};

await test("guardrail bad-word filtering catches a term split across stream deltas", async () => {
  const server = await startSplitDeltaChatServer(SPLIT_TERM);
  const sdk = new NeuroLink();
  try {
    const base = {
      input: { text: "hello" },
      provider: "openai",
      model: "gpt-4o-mini",
      disableTools: true,
      disableInternalFallback: true,
      credentials: {
        openai: { apiKey: "sk-mock-local-server", baseURL: server.baseURL },
      },
    };
    const unfiltered = await bounded(readText(await sdk.stream(base)));
    assert.equal(
      server.requests(),
      1,
      "precondition: split server not exercised",
    );
    assert.equal(
      unfiltered,
      WHOLE_TERM,
      "precondition: split deltas did not reassemble into the whole term",
    );
    const filtered = await bounded(
      readText(await sdk.stream({ ...base, middleware: splitTermGuardrails })),
    );
    assert.equal(
      server.requests(),
      2,
      "guarded stream did not reach the provider",
    );
    assert.equal(
      filtered,
      "CLEAN",
      "term split across stream deltas escaped the guardrail",
    );
  } finally {
    await sdk.shutdown();
    await server.close();
  }
});

await test("guardrail bad-word filtering catches a term split across adjacent text parts (generate)", async () => {
  const server = await startSplitBlockAnthropicServer(SPLIT_TERM);
  try {
    await withEnv(
      {
        ANTHROPIC_BASE_URL: server.baseURL,
        ANTHROPIC_API_KEY: "sk-ant-mock-local-server",
      },
      async () => {
        const sdk = new NeuroLink();
        try {
          const base = {
            input: { text: "hello" },
            provider: "anthropic",
            model: "claude-sonnet-4-20250514",
            disableTools: true,
            disableInternalFallback: true,
          };
          const unfiltered = await sdk.generate(base);
          assert.equal(
            server.requests(),
            1,
            "precondition: split server not exercised",
          );
          assert.equal(
            unfiltered.content,
            WHOLE_TERM,
            "precondition: adjacent text blocks did not reassemble into the whole term",
          );
          const filtered = await sdk.generate({
            ...base,
            middleware: splitTermGuardrails,
          });
          assert.equal(
            server.requests(),
            2,
            "guarded generate did not reach the provider",
          );
          assert.equal(
            filtered.content,
            "CLEAN",
            "term split across adjacent text parts escaped the guardrail",
          );
        } finally {
          await sdk.shutdown();
        }
      },
    );
  } finally {
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// AI STUDIO — native generate()/stream() middleware.
//
// Google AI Studio's generate() and stream() never go through the AI SDK's
// LanguageModel plumbing the way the OpenAI-compatible family above does —
// both hand-roll an agentic loop directly against @google/genai — so before
// this fix neither transformParams nor wrapGenerate/wrapStream ever ran on
// this provider (see docs/plans/2026-09-07-middleware-on-native-providers.md).
// The stand-in below speaks the Gemini REST wire format directly and is
// reached via the public `credentials.googleAiStudio.baseURL` option, the
// same mechanism continuous-test-suite-aistudio-loop-characterization.ts
// uses. Sibling PRs add their own sections here for Vertex and Bedrock.
//
// `createProbe` / `emptyRecord` / `middlewareOptions` / `MARKER` / `readText`
// / `bounded` above are provider-agnostic and reused as-is.
// ---------------------------------------------------------------------------

const AI_STUDIO_MODEL = "gemini-2.0-flash";

/**
 * One streamed candidate chunk in the SSE framing the @google/genai SDK
 * expects. Omitting `finishReason` (undefined, not "STOP") produces a
 * non-terminal chunk — the shape a real mid-turn delta has.
 */
function aiStudioSse(
  parts: Array<Record<string, unknown>>,
  finishReason?: string,
): string {
  const payload = {
    candidates: [
      {
        content: { parts, role: "model" },
        ...(finishReason ? { finishReason } : {}),
        index: 0,
      },
    ],
    usageMetadata: {
      promptTokenCount: 5,
      candidatesTokenCount: 4,
      totalTokenCount: 9,
    },
  };
  return `data: ${JSON.stringify(payload)}\r\n\r\n`;
}

function aiStudioTextTurn(text: string): string {
  return aiStudioSse([{ text }], "STOP");
}

/**
 * A single-turn SSE response carrying cache-read and reasoning (thinking)
 * token counts on `usageMetadata` — the shape Gemini reports when part of
 * the prompt was served from cache and part of the response was thinking.
 * `promptTokenCount` is deliberately cache-inclusive, matching the
 * OVERLAPPING convention documented at its call sites in client.ts.
 */
function aiStudioCacheUsageTurn(text: string): string {
  const payload = {
    candidates: [
      {
        content: { parts: [{ text }], role: "model" },
        finishReason: "STOP",
        index: 0,
      },
    ],
    usageMetadata: {
      promptTokenCount: 1000,
      candidatesTokenCount: 100,
      cachedContentTokenCount: 800,
      thoughtsTokenCount: 50,
      totalTokenCount: 1150,
    },
  };
  return `data: ${JSON.stringify(payload)}\r\n\r\n`;
}

type AiStudioStandInCall = { body: Record<string, unknown> };
type AiStudioStandIn = {
  calls: AiStudioStandInCall[];
  port: number;
  close: () => Promise<void>;
};

async function startAiStudioStandIn(
  reply: (callIndex: number) => string,
): Promise<AiStudioStandIn> {
  const calls: AiStudioStandInCall[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      let body: Record<string, unknown>;
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        body = {};
      }
      calls.push({ body });
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(reply(calls.length - 1));
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return {
    calls,
    port: typeof address === "object" && address ? address.port : 0,
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
      }),
  };
}

/**
 * Opens the response, writes one non-terminal chunk (enough for a consumer
 * to observe content and break out), then stays silent — used by the
 * cancellation case to prove a caller breaking out of the returned stream
 * still reaches the upstream socket through the new V3 wrapping.
 */
async function startAiStudioSilentStandIn(): Promise<{
  received: Promise<void>;
  closed: () => boolean;
  port: number;
  close: () => Promise<void>;
}> {
  let resolveReceived: () => void;
  const received = new Promise<void>((resolve) => {
    resolveReceived = resolve;
  });
  let closed = false;
  const server: Server = createServer((req, res) => {
    req.on("data", () => {
      /* drain */
    });
    req.on("end", () => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.on("close", () => {
        closed = true;
      });
      res.write(aiStudioSse([{ text: "first" }]));
      resolveReceived();
      // Deliberately never ends — only cancellation should close this.
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return {
    received,
    closed: () => closed,
    port: typeof address === "object" && address ? address.port : 0,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections?.();
        server.close(() => resolve());
      }),
  };
}

function aiStudioCredentialsFor(port: number) {
  return {
    googleAiStudio: {
      apiKey: "test-key",
      baseURL: `http://127.0.0.1:${port}`,
    },
  };
}

function lastAiStudioBody(
  server: AiStudioStandIn,
): Record<string, unknown> | undefined {
  return server.calls[server.calls.length - 1]?.body;
}

await test("AI Studio: generate applies model middleware (precondition)", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    assert.ok(
      record.transformParamsCalls.length > 0,
      "probe never ran on AI Studio generate — the native generate path bypasses caller model middleware",
    );
    assert.ok(
      record.wrapGenerateCalls > 0,
      "wrapGenerate never fired on the AI Studio generate path",
    );
    const body = lastAiStudioBody(server);
    assert.ok(
      body && JSON.stringify(body.contents ?? {}).includes(MARKER),
      "the transformParams rewrite did not reach the wire on AI Studio generate",
    );
  } finally {
    await server.close();
  }
});

await test("AI Studio: stream applies model middleware", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    const result = await nl.stream({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    await bounded(readText(result));
    assert.ok(
      server.calls.length > 0,
      "the AI Studio stand-in was never called — the stream never left the machine, so nothing below can be concluded about middleware",
    );
    assert.ok(
      record.transformParamsCalls.length > 0,
      "transformParams never fired on the AI Studio streaming path",
    );
    assert.ok(
      record.transformParamsCalls.includes("stream"),
      'transformParams fired but never with type "stream" on AI Studio stream()',
    );
    assert.ok(
      record.wrapStreamCalls > 0,
      "wrapStream never fired on the AI Studio streaming path",
    );
  } finally {
    await server.close();
  }
});

await test("AI Studio: a stream transformParams rewrite reaches the wire", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const record = emptyRecord();
  try {
    const nl = new NeuroLink();
    const result = await nl.stream({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    await bounded(readText(result));
    const body = lastAiStudioBody(server);
    assert.ok(
      body,
      "the AI Studio stand-in captured no request body for the stream",
    );
    assert.ok(
      JSON.stringify(body?.contents ?? {}).includes(MARKER),
      "the transformParams rewrite did not reach the wire on AI Studio stream",
    );
  } finally {
    await server.close();
  }
});

await test("AI Studio: wrapStream observes the V3 terminal event with usage", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  const seen: LanguageModelV3StreamPart[] = [];
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "ai-studio-wire-filter", name: "AI Studio wire filter" },
    wrapStream: async ({ doStream }) => {
      const result = await doStream();
      return {
        ...result,
        stream: result.stream.pipeThrough(
          new TransformStream({
            transform(part: LanguageModelV3StreamPart, controller) {
              seen.push(part);
              controller.enqueue(
                part.type === "text-delta"
                  ? { ...part, delta: part.delta.toUpperCase() }
                  : part,
              );
            },
          }),
        ),
      };
    },
  };
  try {
    const result = await nl.stream({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-wire-filter"],
      },
    });
    const text = await bounded(readText(result));
    assert.equal(
      text,
      "HELLO FROM AI STUDIO",
      "filtered text lost on AI Studio stream",
    );
    assert.ok(server.calls.length > 0, "AI Studio stand-in was never reached");
    const finishParts = seen.filter((part) => part.type === "finish");
    assert.equal(
      finishParts.length,
      1,
      "terminal event not forwarded on AI Studio stream",
    );
    const finish = finishParts[0];
    assert.ok(
      finish.type === "finish" && (finish.usage.outputTokens.total ?? 0) > 0,
      "AI Studio finish part carried no usage",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: wrapGenerate observes the V3 generate result with usage", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  let observedText: string | undefined;
  let observedOutputTokens: number | undefined;
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "ai-studio-generate-observer",
      name: "AI Studio generate observer",
    },
    wrapGenerate: async ({ doGenerate }) => {
      const result = await doGenerate();
      const textPart = result.content.find(
        (c): c is { type: "text"; text: string } => c.type === "text",
      );
      observedText = textPart?.text;
      observedOutputTokens = result.usage.outputTokens.total;
      return result;
    },
  };
  try {
    const result = await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-generate-observer"],
      },
    });
    assert.equal(
      result.content,
      "hello from ai studio",
      "AI Studio generate content lost through middleware",
    );
    assert.ok(
      observedText?.includes("hello from ai studio"),
      "wrapGenerate never observed the V3 text content on AI Studio generate",
    );
    assert.ok(
      (observedOutputTokens ?? 0) > 0,
      "wrapGenerate observed a V3 result with no usage on AI Studio generate",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

// `gemini-3-fable-preview` isn't a registered model, so it falls back to
// `SAMPLING_PARAM_REJECTING_FAMILIES` in modelRegistry.ts, which matches it
// on `/fable/i` — the shared cross-provider family that rejects classic
// sampling params. `buildNativeConfig` strips temperature/topP for it; the
// generate bridge must not reintroduce them from the raw call options.
const SAMPLING_REJECTING_MODEL = "gemini-3-fable-preview";

await test("AI Studio: generate keeps the sampling-param strip in effect for a model that rejects them", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  try {
    const base = {
      input: { text: "hi" },
      provider: "google-ai",
      disableTools: true,
      disableInternalFallback: true,
      maxTokens: 50,
      temperature: 0.42,
      topP: 0.77,
      credentials: aiStudioCredentialsFor(server.port),
    };

    // Control: a model the registry treats as supporting sampling params —
    // establishes that this harness's request/response/body-capture round
    // trip actually carries temperature/topP end to end, so an absence on
    // the rejecting-family run below means the strip held, not that the
    // field never makes it onto the wire in this test at all.
    await nl.generate({ ...base, model: AI_STUDIO_MODEL });
    assert.equal(
      server.calls.length,
      1,
      "precondition: AI Studio generate control call not reached",
    );
    const controlBody = lastAiStudioBody(server) as {
      generationConfig?: { temperature?: number; topP?: number };
    };
    assert.equal(
      controlBody.generationConfig?.temperature,
      0.42,
      "precondition: control model lost its temperature",
    );
    assert.equal(
      controlBody.generationConfig?.topP,
      0.77,
      "precondition: control model lost its topP",
    );

    // Test: same call options, a sampling-rejecting model id.
    await nl.generate({ ...base, model: SAMPLING_REJECTING_MODEL });
    assert.equal(
      server.calls.length,
      2,
      "precondition: AI Studio generate rejecting-family call not reached",
    );
    const strippedBody = lastAiStudioBody(server) as {
      generationConfig?: {
        temperature?: number;
        topP?: number;
        maxOutputTokens?: number;
      };
    };
    // An ungated field (never subject to the sampling-param strip) proves
    // this body was parsed and inspected correctly, not merely empty.
    assert.equal(
      strippedBody.generationConfig?.maxOutputTokens,
      50,
      "precondition: rejecting-family generate body missing an ungated field",
    );
    assert.equal(
      strippedBody.generationConfig?.temperature,
      undefined,
      "AI Studio generate reintroduced temperature for a sampling-rejecting model",
    );
    assert.equal(
      strippedBody.generationConfig?.topP,
      undefined,
      "AI Studio generate reintroduced topP for a sampling-rejecting model",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: stream middleware sampling edits reach the wire", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "ai-studio-sampling", name: "AI Studio sampling" },
    transformParams: async ({ params }) => ({
      ...params,
      maxOutputTokens: 77,
      temperature: 0.25,
      topP: 0.9,
    }),
  };
  try {
    await bounded(
      readText(
        await nl.stream({
          input: { text: "hi" },
          provider: "google-ai",
          model: AI_STUDIO_MODEL,
          disableTools: true,
          disableInternalFallback: true,
          maxTokens: 128,
          temperature: 0.7,
          credentials: aiStudioCredentialsFor(server.port),
          middleware: {
            middleware: [middleware],
            enabledMiddleware: ["ai-studio-sampling"],
          },
        }),
      ),
    );
    assert.ok(
      server.calls.length > 0,
      "AI Studio sampling fixture not reached",
    );
    const body = lastAiStudioBody(server) as {
      generationConfig?: {
        temperature?: number;
        maxOutputTokens?: number;
        topP?: number;
      };
    };
    assert.equal(
      body.generationConfig?.maxOutputTokens,
      77,
      "AI Studio token override lost",
    );
    assert.equal(
      body.generationConfig?.temperature,
      0.25,
      "AI Studio temperature override lost",
    );
    assert.equal(
      body.generationConfig?.topP,
      0.9,
      "AI Studio top-p override lost",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: stream keeps the sampling-param strip in effect for a model that rejects them", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  try {
    const base = {
      input: { text: "hi" },
      provider: "google-ai",
      disableTools: true,
      disableInternalFallback: true,
      maxTokens: 50,
      temperature: 0.42,
      topP: 0.77,
      credentials: aiStudioCredentialsFor(server.port),
    };

    // Control: a model the registry treats as supporting sampling params —
    // establishes that this harness's request/response/body-capture round
    // trip actually carries temperature/topP end to end, so an absence on
    // the rejecting-family run below means the strip held, not that the
    // field never makes it onto the wire in this test at all.
    await bounded(
      readText(await nl.stream({ ...base, model: AI_STUDIO_MODEL })),
    );
    assert.equal(
      server.calls.length,
      1,
      "precondition: AI Studio stream control call not reached",
    );
    const controlBody = lastAiStudioBody(server) as {
      generationConfig?: { temperature?: number; topP?: number };
    };
    assert.equal(
      controlBody.generationConfig?.temperature,
      0.42,
      "precondition: control model lost its temperature",
    );
    assert.equal(
      controlBody.generationConfig?.topP,
      0.77,
      "precondition: control model lost its topP",
    );

    // Test: same call options, a sampling-rejecting model id (see
    // SAMPLING_REJECTING_MODEL's definition above the generate-side twin
    // of this test for why `gemini-3-fable-preview` matches the family).
    await bounded(
      readText(await nl.stream({ ...base, model: SAMPLING_REJECTING_MODEL })),
    );
    assert.equal(
      server.calls.length,
      2,
      "precondition: AI Studio stream rejecting-family call not reached",
    );
    const strippedBody = lastAiStudioBody(server) as {
      generationConfig?: {
        temperature?: number;
        topP?: number;
        maxOutputTokens?: number;
      };
    };
    // An ungated field (never subject to the sampling-param strip) proves
    // this body was parsed and inspected correctly, not merely empty.
    assert.equal(
      strippedBody.generationConfig?.maxOutputTokens,
      50,
      "precondition: rejecting-family stream body missing an ungated field",
    );
    assert.equal(
      strippedBody.generationConfig?.temperature,
      undefined,
      "AI Studio stream reintroduced temperature for a sampling-rejecting model",
    );
    assert.equal(
      strippedBody.generationConfig?.topP,
      undefined,
      "AI Studio stream reintroduced topP for a sampling-rejecting model",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

for (const mode of ["generate", "stream"] as const) {
  await test(`AI Studio: precall guardrail blocks ${mode} — target never called`, async () => {
    const evaluator = await startScriptedChatServer([
      chatCompletion({
        content: JSON.stringify({
          overall: "unsafe",
          safetyScore: 1,
          appropriatenessScore: 1,
          confidenceLevel: 10,
          suggestedAction: "block",
          reasoning: "Deterministic blocking fixture",
        }),
      }),
    ]);
    const target = await startAiStudioStandIn(() =>
      aiStudioTextTurn("should never be seen"),
    );
    const saved = {
      key: process.env.OPENAI_COMPATIBLE_API_KEY,
      url: process.env.OPENAI_COMPATIBLE_BASE_URL,
    };
    process.env.OPENAI_COMPATIBLE_API_KEY = "test-evaluator-key";
    process.env.OPENAI_COMPATIBLE_BASE_URL = evaluator.baseURL;
    const nl = new NeuroLink();
    try {
      const options = {
        input: { text: "block this request" },
        provider: "google-ai",
        model: AI_STUDIO_MODEL,
        disableTools: true,
        disableInternalFallback: true,
        enableAnalytics: true,
        credentials: aiStudioCredentialsFor(target.port),
        middleware: {
          middlewareConfig: {
            guardrails: {
              enabled: true,
              config: {
                precallEvaluation: {
                  enabled: true,
                  provider: "openai-compatible",
                  evaluationModel: "fixture-evaluator",
                },
              },
            },
          },
        },
      };
      const result =
        mode === "generate"
          ? await nl.generate(options)
          : await nl.stream(options);
      const content =
        result && "stream" in result
          ? await bounded(readText(result))
          : result?.content;
      assert.ok(evaluator.wasCalled(), "guardrail evaluator was not exercised");
      assert.equal(
        target.calls.length,
        0,
        "blocked input reached the AI Studio target",
      );
      assert.equal(
        content,
        "Request contains inappropriate content and has been blocked.",
        "AI Studio guardrail refusal was lost",
      );
      if (result && "analytics" in result && result.analytics) {
        await bounded(Promise.resolve(result.analytics));
      }
    } finally {
      if (saved.key === undefined) {
        delete process.env.OPENAI_COMPATIBLE_API_KEY;
      } else {
        process.env.OPENAI_COMPATIBLE_API_KEY = saved.key;
      }
      if (saved.url === undefined) {
        delete process.env.OPENAI_COMPATIBLE_BASE_URL;
      } else {
        process.env.OPENAI_COMPATIBLE_BASE_URL = saved.url;
      }
      await nl.shutdown();
      await target.close();
      await evaluator.close();
    }
  });
}

await test("AI Studio: breaking out of a wrapped stream cancels the upstream socket", async () => {
  const standIn = await startAiStudioSilentStandIn();
  const nl = new NeuroLink();
  try {
    const result = await nl.stream({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(standIn.port),
      middleware: middlewareOptions(emptyRecord()),
    });
    await bounded(
      (async () => {
        for await (const chunk of result.stream) {
          if ("content" in chunk && chunk.content) {
            break;
          }
        }
      })(),
    );
    await bounded(standIn.received);
    await bounded(
      (async () => {
        while (!standIn.closed()) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      })(),
    );
    assert.ok(
      standIn.closed(),
      "upstream socket not closed on AI Studio cancellation",
    );
  } finally {
    await nl.shutdown();
    await standIn.close();
  }
});

await test("AI Studio: audio input bypasses caller model middleware entirely", async () => {
  // A probe that accepts the raw TCP connection and immediately destroys
  // the socket. Nothing ever completes the WebSocket handshake, so the
  // request never succeeds — but critically, @google/genai's Live client
  // (dist/node/index.cjs Live.connect) only settles its internal
  // `onopenPromise` from the `onopen` callback; the default/no-op `onerror`
  // path it wires up does NOT reject that promise. So this branch does not
  // fail fast with ECONNREFUSED the way a plain HTTP request would —
  // verified empirically, it hangs until something outside the SDK gives
  // up (here, the `bounded` wrapper below). That rules out asserting on any
  // error/message the audio dispatch produces, since none reliably arrives.
  // The one signal that is real: the probe itself observing a raw TCP
  // connection, which only happens once the audio/Gemini Live branch has
  // actually dialed out. That is the precondition below — the absence of
  // middleware activity proves nothing about *this* branch unless the
  // branch is proven to have been dispatched first.
  let connectionsSeen = 0;
  const probe = createServer();
  probe.on("connection", (socket) => {
    connectionsSeen += 1;
    socket.destroy();
  });
  const probePort = await new Promise<number>((resolve) => {
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address ? address.port : 0;
      resolve(port);
    });
  });

  async function* silentFrame(): AsyncIterable<Buffer> {
    yield Buffer.alloc(320);
  }

  const record = emptyRecord();
  const nl = new NeuroLink();
  try {
    try {
      // 10s, not the suite's usual 5s: the observed dial-out latency to
      // this probe (dynamic `@google/genai` import + Live handshake attempt)
      // runs ~3-5.5s before the socket is even opened, and the branch never
      // settles on its own (see above) — so the budget only needs to
      // outlast that dial-out, not a real response.
      await bounded(
        nl.stream({
          input: { audio: { frames: silentFrame() } },
          provider: "google-ai",
          model: "gemini-2.5-flash-preview-native-audio-dialog",
          disableInternalFallback: true,
          credentials: aiStudioCredentialsFor(probePort),
          middleware: middlewareOptions(record),
        }),
        10000,
      );
    } catch {
      // Expected — the probe never completes the handshake. Only the
      // assertions below are under test.
    }
    // Precondition: prove the audio/Gemini Live branch was actually
    // dispatched before trusting the negative assertions below. Without
    // this, a caller-model-middleware regression that instead sent the
    // request down the ordinary text/tool path — and failed for some
    // unrelated reason before ever touching transformParams/wrapStream —
    // would satisfy both negative assertions for the wrong reason. This
    // checks the transport-level fact (a TCP connection reached the probe),
    // not any error message, since this branch does not reliably produce one.
    assert.ok(
      connectionsSeen > 0,
      "the audio/Gemini Live branch was never dispatched — the probe observed no connection attempt, so the assertions below would prove nothing",
    );
    assert.equal(
      record.transformParamsCalls.length,
      0,
      "transformParams fired on the audio (Gemini Live) branch",
    );
    assert.equal(
      record.wrapStreamCalls,
      0,
      "wrapStream fired on the audio (Gemini Live) branch",
    );
  } finally {
    await nl.shutdown();
    await new Promise<void>((resolve) => probe.close(() => resolve()));
  }
});

// ---------------------------------------------------------------------------
// AI STUDIO — caller model middleware system-prompt visibility (forward and
// reverse), short-circuit stream cleanup, and native-turn usage fidelity.
// Gemini has no wire-visible system role: the instruction rides separately
// on `config.systemInstruction`, never inside `contents`. Before the fix,
// `geminiContentsToV3Prompt(contents)` alone fed `transformParams`, so the
// caller's system prompt was invisible to middleware on both bridges, and
// the reverse direction only ever overrode `systemInstruction` when
// `systemText` was truthy — never signaling "the caller removed it".
// ---------------------------------------------------------------------------

await test("AI Studio: transformParams observes the caller's system prompt on generate", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  let observedSystemContents: string[] = [];
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "ai-studio-system-observer",
      name: "AI Studio system observer",
    },
    transformParams: async ({ params }) => {
      observedSystemContents = params.prompt
        .filter((message) => message.role === "system")
        .map((message) => message.content);
      return params;
    },
  };
  const nl = new NeuroLink();
  try {
    await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      systemPrompt: "Answer only in French.",
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-system-observer"],
      },
    });
    assert.ok(server.calls.length > 0, "AI Studio stand-in was never reached");
    assert.equal(
      observedSystemContents.length,
      1,
      "transformParams did not see a system message for the caller's systemPrompt on AI Studio generate",
    );
    assert.equal(
      observedSystemContents[0],
      "Answer only in French.",
      "the system message transformParams observed did not carry the caller's systemPrompt text",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: a middleware-added system message composes with the caller's on the wire", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "ai-studio-system-append",
      name: "AI Studio system append",
    },
    transformParams: async ({ params }) => ({
      ...params,
      prompt: [
        {
          role: "system" as const,
          content: "Always answer in bullet points.",
        },
        ...params.prompt,
      ],
    }),
  };
  const nl = new NeuroLink();
  try {
    await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      systemPrompt: "Answer only in French.",
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-system-append"],
      },
    });
    const body = lastAiStudioBody(server) as { systemInstruction?: unknown };
    assert.ok(
      body?.systemInstruction,
      "AI Studio wire request carried no systemInstruction at all",
    );
    const wireSystemText = JSON.stringify(body.systemInstruction);
    assert.ok(
      wireSystemText.includes("Always answer in bullet points."),
      "the middleware-added system message never reached the AI Studio wire request",
    );
    assert.ok(
      wireSystemText.includes("Answer only in French."),
      "the caller's original systemPrompt was clobbered by the middleware-added system message",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: middleware clearing every system message removes systemInstruction from the wire", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("hello from ai studio"),
  );
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "ai-studio-system-strip", name: "AI Studio system strip" },
    transformParams: async ({ params }) => ({
      ...params,
      prompt: params.prompt.filter((message) => message.role !== "system"),
    }),
  };
  const nl = new NeuroLink();
  try {
    await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      systemPrompt: "Answer only in French.",
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-system-strip"],
      },
    });
    assert.ok(server.calls.length > 0, "AI Studio stand-in was never reached");
    const body = lastAiStudioBody(server) as { systemInstruction?: unknown };
    assert.equal(
      body?.systemInstruction,
      undefined,
      "middleware removed every system message but the caller's systemPrompt still reached the AI Studio wire request",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: analytics settle when a short-circuiting stream is abandoned before finish", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioTextTurn("should never be reached"),
  );
  const middleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "ai-studio-short-circuit",
      name: "AI Studio short circuit",
    },
    wrapStream: async () => ({
      stream: new ReadableStream<LanguageModelV3StreamPart>({
        start(controller) {
          controller.enqueue({ type: "text-start", id: "sc" });
          controller.enqueue({
            type: "text-delta",
            id: "sc",
            delta: "partial",
          });
          // Deliberately never enqueues "finish" and never closes — models
          // a short-circuit stream a caller abandons mid-read, the case
          // the pre-fix cleanup (gated only on a "finish" part) never ran.
        },
      }),
    }),
  };
  const nl = new NeuroLink();
  try {
    const result = await nl.stream({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      enableAnalytics: true,
      credentials: aiStudioCredentialsFor(server.port),
      middleware: {
        middleware: [middleware],
        enabledMiddleware: ["ai-studio-short-circuit"],
      },
    });
    for await (const chunk of result.stream) {
      if ("content" in chunk && chunk.content) {
        break;
      }
    }
    assert.equal(
      server.calls.length,
      0,
      "synthetic short-circuit stream reached the AI Studio wire — the wrapStream short-circuit did not take effect",
    );
    assert.ok(
      result.analytics,
      "no analytics promise was exposed on the short-circuit stream",
    );
    const analytics = await bounded(Promise.resolve(result.analytics), 5000);
    assert.ok(
      (analytics as AnalyticsData).stopReason,
      "abandoned short-circuit stream settled analytics with no stopReason",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

await test("AI Studio: generate usage preserves cache-read and reasoning tokens from the native turn", async () => {
  const server = await startAiStudioStandIn(() =>
    aiStudioCacheUsageTurn("hello from ai studio"),
  );
  const nl = new NeuroLink();
  try {
    const result = await nl.generate({
      input: { text: "hi" },
      provider: "google-ai",
      model: AI_STUDIO_MODEL,
      disableTools: true,
      disableInternalFallback: true,
      credentials: aiStudioCredentialsFor(server.port),
    });
    assert.ok(server.calls.length > 0, "AI Studio stand-in was never reached");
    assert.equal(
      result.content,
      "hello from ai studio",
      "AI Studio generate content lost while exercising cache/reasoning usage",
    );
    assert.equal(
      result.usage?.cacheReadTokens,
      800,
      "cache-read tokens from the native AI Studio turn were dropped from the generate result's usage",
    );
    assert.equal(
      result.usage?.reasoning,
      50,
      "reasoning tokens from the native AI Studio turn were dropped from the generate result's usage",
    );
    assert.equal(
      result.usage?.total,
      1150,
      "the generate result's usage total undercounted once cache/reasoning tokens were introduced",
    );
  } finally {
    await nl.shutdown();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// VERTEX — native generate()/stream() loops (Item E4b).
//
// AI Studio and Bedrock get their own sections in separate PRs; this one is
// Vertex-only. Vertex has FOUR native loops — Gemini3 and Anthropic-on-Vertex,
// each with its own generate() and stream() — that used to call
// executeNative*Generate/executeNative*Stream directly, bypassing
// `applyMiddlewareToModel` entirely: the `middleware` option was accepted and
// silently had no effect. `runNativeGenerateWithMiddleware` /
// `runNativeStreamWithMiddleware` in googleVertex/client.ts now wrap every one
// of the four with the same `wrapLanguageModel` chain every other provider
// already goes through.
//
// Reaching any of this offline needs Vertex AI Express Mode
// (`credentials.vertex.apiKey` with no project/location, which skips ADC and
// honours `credentials.vertex.baseURL`) — the same affordance
// vertex-loop-characterization.ts and vertex-claude-characterization.ts rely
// on. Neither file exports its helpers, so the pieces needed here (env
// clearing, the two wire formats' SSE builders, the two stand-ins) are
// re-implemented locally rather than imported.
// ---------------------------------------------------------------------------

section("Vertex");

const GEMINI_MODEL = "gemini-2.0-flash";
const ANTHROPIC_MODEL = "claude-3-5-sonnet-v2@20241022";

/**
 * Every env var that could pull the provider onto the ADC path instead of
 * Express Mode. All four project-name fallbacks matter — clearing only one
 * leaves an ambient project on a dev machine hiding the very path under test.
 */
const VERTEX_TOUCHED_ENV_VARS = [
  "GOOGLE_CLOUD_PROJECT",
  "GOOGLE_CLOUD_PROJECT_ID",
  "VERTEX_PROJECT_ID",
  "GOOGLE_VERTEX_PROJECT",
  "GOOGLE_CLOUD_LOCATION",
  "VERTEX_LOCATION",
  "GOOGLE_VERTEX_LOCATION",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "GOOGLE_VERTEX_API_KEY",
  "GOOGLE_VERTEX_BASE_URL",
  "GOOGLE_API_KEY",
] as const;

function withVertexEnv(): () => void {
  const saved: Record<string, string | undefined> = {};
  for (const key of VERTEX_TOUCHED_ENV_VARS) {
    saved[key] = process.env[key];
    delete process.env[key];
  }
  return () => {
    for (const key of VERTEX_TOUCHED_ENV_VARS) {
      const prior = saved[key];
      if (prior === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = prior;
      }
    }
  };
}

function vertexCredentialsFor(port: number) {
  return {
    vertex: { apiKey: "express-key", baseURL: `http://127.0.0.1:${port}` },
  };
}

type VertexCapturedFinish = {
  finishReasonUnified: string;
  inputTokens: number | undefined;
  outputTokens: number | undefined;
};

// --- Gemini3 wire: Gemini `generateContentStream` SSE framing --------------

function geminiSse(
  parts: Array<Record<string, unknown>>,
  finishReason?: string,
): string {
  const payload = {
    candidates: [
      {
        content: { parts, role: "model" },
        ...(finishReason ? { finishReason } : {}),
        index: 0,
      },
    ],
    usageMetadata: {
      promptTokenCount: 5,
      candidatesTokenCount: 4,
      totalTokenCount: 9,
    },
  };
  return `data: ${JSON.stringify(payload)}\r\n\r\n`;
}

function geminiTextTurn(text: string): string {
  return geminiSse([{ text }], "STOP");
}

type GeminiStandInCall = { body: Record<string, unknown> };
type GeminiStandIn = {
  calls: GeminiStandInCall[];
  port: number;
  close: () => Promise<void>;
};

/** An SSE body, or a JSON error the stand-in answers with a non-200 status. */
type GeminiStandInReply = string | { status: number; json: unknown };

async function startGeminiStandIn(
  reply: (callIndex: number) => GeminiStandInReply,
): Promise<GeminiStandIn> {
  const calls: GeminiStandInCall[] = [];
  const httpServer: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        // Malformed body — keep it empty rather than fail the request.
      }
      calls.push({ body });
      const answer = reply(calls.length - 1);
      if (typeof answer === "string") {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(answer);
        res.end();
        return;
      }
      res.writeHead(answer.status, { "content-type": "application/json" });
      res.end(JSON.stringify(answer.json));
    });
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  return {
    calls,
    port: typeof address === "object" && address ? address.port : 0,
    close: () =>
      new Promise<void>((resolve) => httpServer.close(() => resolve())),
  };
}

// --- Anthropic-on-Vertex wire: Anthropic Messages SSE framing --------------

function anthropicSse(event: string, payload: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify({ type: event, ...payload })}\n\n`;
}

/** A turn that emits text and stops — the full message shape, since
 * `MessageStream` builds its running snapshot from `message_start` and dies
 * pushing onto `snapshot.content` without it. */
function anthropicTextTurn(text: string): string[] {
  return [
    anthropicSse("message_start", {
      message: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: ANTHROPIC_MODEL,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 5, output_tokens: 0 },
      },
    }),
    anthropicSse("content_block_start", {
      index: 0,
      content_block: { type: "text", text: "" },
    }),
    anthropicSse("content_block_delta", {
      index: 0,
      delta: { type: "text_delta", text },
    }),
    anthropicSse("content_block_stop", { index: 0 }),
    anthropicSse("message_delta", {
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 4 },
    }),
    anthropicSse("message_stop", {}),
  ];
}

type AnthropicStandInCall = { body: Record<string, unknown> };
type AnthropicStandIn = {
  calls: AnthropicStandInCall[];
  port: number;
  close: () => Promise<void>;
};

/**
 * Folds a case's SSE frames into the single JSON message the non-streaming
 * endpoint returns, so one fixture serves both wire modes:
 * executeNativeAnthropicStream calls `messages.stream` (SSE), while
 * executeNativeAnthropicGenerate calls `messages.create` (single JSON body).
 */
function anthropicMessageFromFrames(frames: string[]): Record<string, unknown> {
  const events = frames.map(
    (frame) =>
      JSON.parse(frame.slice(frame.indexOf("data: ") + 6).trim()) as Record<
        string,
        unknown
      >,
  );
  const start = events.find((e) => e.type === "message_start") as
    | { message?: Record<string, unknown> }
    | undefined;
  const message: Record<string, unknown> = { ...(start?.message ?? {}) };
  const content: Array<Record<string, unknown>> = [];
  for (const event of events) {
    if (event.type === "content_block_start") {
      content.push({
        ...((event.content_block as Record<string, unknown>) ?? {}),
      });
    }
    if (event.type === "content_block_delta") {
      const delta = (event.delta ?? {}) as Record<string, unknown>;
      const block = content[content.length - 1];
      if (!block) {
        continue;
      }
      if (typeof delta.text === "string") {
        block.text = `${block.text ?? ""}${delta.text}`;
      }
    }
    if (event.type === "message_delta") {
      Object.assign(message, (event.delta ?? {}) as Record<string, unknown>);
    }
  }
  message.content = content;
  return message;
}

async function startAnthropicStandIn(
  reply: (callIndex: number) => string[],
): Promise<AnthropicStandIn> {
  const calls: AnthropicStandInCall[] = [];
  const httpServer: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        // Malformed body — keep it empty rather than fail the request.
      }
      calls.push({ body });
      const frames = reply(calls.length - 1);
      if (body.stream === true) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        for (const frame of frames) {
          res.write(frame);
        }
        res.end();
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(anthropicMessageFromFrames(frames)));
    });
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  return {
    calls,
    port: typeof address === "object" && address ? address.port : 0,
    close: () =>
      new Promise<void>((resolve) => httpServer.close(() => resolve())),
  };
}

/**
 * Opens with the first frames of a turn and then never finishes the response
 * — the upstream a consumer walks away from mid-stream. `closedResponses()`
 * counts responses whose connection ended. Nothing here ever ends one, so a
 * client abort is the only thing that can raise it: it is the evidence that
 * the call was released rather than merely abandoned.
 */
type AnthropicHoldOpenStandIn = AnthropicStandIn & {
  closedResponses: () => number;
};

async function startAnthropicHoldOpenStandIn(): Promise<AnthropicHoldOpenStandIn> {
  const calls: AnthropicStandInCall[] = [];
  let closed = 0;
  const httpServer: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        // Malformed body — keep it empty rather than fail the request.
      }
      calls.push({ body });
      res.on("close", () => {
        closed += 1;
      });
      res.writeHead(200, { "content-type": "text/event-stream" });
      // message_start, content_block_start, one text delta — then silence.
      for (const frame of anthropicTextTurn("held-open reply").slice(0, 3)) {
        res.write(frame);
      }
    });
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  return {
    calls,
    port: typeof address === "object" && address ? address.port : 0,
    closedResponses: () => closed,
    close: () => {
      httpServer.closeAllConnections();
      return new Promise<void>((resolve) => httpServer.close(() => resolve()));
    },
  };
}

// --- What one upstream request carried, read through each wire format ------
//
// The two formats name the same things differently, so a case that asserts on
// "the system prompt" or "the sampling params" reads them through the loop's
// own reader instead of searching the serialized body for a string. A body
// search cannot tell the system prompt from the user turn.

type VertexWire = {
  system: string | undefined;
  userTexts: string[];
  temperature: number | undefined;
  topP: number | undefined;
  maxTokens: number | undefined;
};

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

const asNumber = (value: unknown): number | undefined =>
  typeof value === "number" ? value : undefined;

/** The `text` of every entry in a parts array, or the string itself. */
const textsOf = (parts: unknown): string[] => {
  if (typeof parts === "string") {
    return [parts];
  }
  return Array.isArray(parts)
    ? parts.flatMap((part) => {
        const text = asRecord(part).text;
        return typeof text === "string" ? [text] : [];
      })
    : [];
};

const joinedOrUndefined = (texts: string[]): string | undefined =>
  texts.length > 0 ? texts.join("\n") : undefined;

const entriesWithRole = (entries: unknown, role: string): unknown[] =>
  (Array.isArray(entries) ? entries : []).filter(
    (entry) => asRecord(entry).role === role,
  );

function geminiWire(body: Record<string, unknown>): VertexWire {
  const config = asRecord(body.generationConfig);
  return {
    system: joinedOrUndefined(textsOf(asRecord(body.systemInstruction).parts)),
    userTexts: entriesWithRole(body.contents, "user").flatMap((entry) =>
      textsOf(asRecord(entry).parts),
    ),
    temperature: asNumber(config.temperature),
    topP: asNumber(config.topP),
    maxTokens: asNumber(config.maxOutputTokens),
  };
}

function anthropicWire(body: Record<string, unknown>): VertexWire {
  return {
    system: joinedOrUndefined(textsOf(body.system)),
    userTexts: entriesWithRole(body.messages, "user").flatMap((entry) =>
      textsOf(asRecord(entry).content),
    ),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    maxTokens: asNumber(body.max_tokens),
  };
}

// --- One case per native loop -----------------------------------------------

type VertexStandIn = {
  calls: Array<{ body: Record<string, unknown> }>;
  port: number;
  close: () => Promise<void>;
};

type VertexLoop = {
  label: string;
  mode: "generate" | "stream";
  model: string;
  start: (reply: string) => Promise<VertexStandIn>;
  wire: (body: Record<string, unknown>) => VertexWire;
};

const VERTEX_REPLY = "vertex reply text";
const USER_TEXT = "ORIGINAL_USER_QUESTION";
const SYSTEM_TEXT = "ORIGINAL_SYSTEM_RULES";

const VERTEX_LOOPS: readonly VertexLoop[] = [
  {
    label: "Gemini native generate",
    mode: "generate",
    model: GEMINI_MODEL,
    start: (reply) => startGeminiStandIn(() => geminiTextTurn(reply)),
    wire: geminiWire,
  },
  {
    label: "Gemini native stream",
    mode: "stream",
    model: GEMINI_MODEL,
    start: (reply) => startGeminiStandIn(() => geminiTextTurn(reply)),
    wire: geminiWire,
  },
  {
    label: "Anthropic native generate",
    mode: "generate",
    model: ANTHROPIC_MODEL,
    start: (reply) => startAnthropicStandIn(() => anthropicTextTurn(reply)),
    wire: anthropicWire,
  },
  {
    label: "Anthropic native stream",
    mode: "stream",
    model: ANTHROPIC_MODEL,
    start: (reply) => startAnthropicStandIn(() => anthropicTextTurn(reply)),
    wire: anthropicWire,
  },
];
const GENERATE_LOOPS = VERTEX_LOOPS.filter((loop) => loop.mode === "generate");
const STREAM_LOOPS = VERTEX_LOOPS.filter((loop) => loop.mode === "stream");

type VertexCallExtras = {
  systemPrompt?: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  enableAnalytics?: boolean;
  middleware?: MiddlewareFactoryOptions;
  onFinish?: OnFinishCallback;
  onError?: OnErrorCallback;
  onChunk?: OnChunkCallback;
};

type VertexRun = {
  text: string;
  analytics: AnalyticsData | undefined;
  bodies: Array<Record<string, unknown>>;
};

/**
 * Lifecycle callbacks are invoked through promise wrappers with deadline
 * timers, so a callback that fires late needs time to show up before a count
 * is read as final. The floor assertion beside every count proves the
 * callbacks were live; this window is what gives the ceilings their meaning.
 */
const settleLifecycle = (): Promise<void> =>
  new Promise<void>((resolve) => setTimeout(resolve, 250));

/** Runs one call on `loop` against a fresh stand-in and reports what it saw. */
async function runVertexLoop(
  loop: VertexLoop,
  extras: VertexCallExtras = {},
): Promise<VertexRun> {
  const server = await loop.start(VERTEX_REPLY);
  const restoreEnv = withVertexEnv();
  const sdk = new NeuroLink();
  try {
    const options = {
      input: { text: USER_TEXT },
      provider: "vertex" as const,
      model: loop.model,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      ...extras,
    };
    if (loop.mode === "generate") {
      const result = await sdk.generate(options);
      await settleLifecycle();
      return {
        text: result.content,
        analytics: result.analytics,
        bodies: server.calls.map((call) => call.body),
      };
    }
    const result = await sdk.stream(options);
    const text = await bounded(readText(result));
    const analytics = await bounded(Promise.resolve(result.analytics));
    await settleLifecycle();
    return {
      text,
      analytics,
      bodies: server.calls.map((call) => call.body),
    };
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
}

/**
 * The first upstream request, read through the loop's wire format. Throws
 * when none arrived, so an assertion about what a request lacks can never
 * pass on a request that was never sent.
 */
function wireOf(loop: VertexLoop, run: VertexRun): VertexWire {
  const body = run.bodies[0];
  if (!body) {
    throw new Error(`the ${loop.label} stand-in was never reached`);
  }
  return loop.wire(body);
}

type MiddlewareParams = Parameters<
  NonNullable<NeuroLinkMiddleware["transformParams"]>
>[0]["params"];
type V3Prompt = MiddlewareParams["prompt"];
type V3UserMessage = Extract<V3Prompt[number], { role: "user" }>;
type V3UserParts = Exclude<V3UserMessage["content"], string>;

/**
 * Prompt content as a middleware written against the V3 spec sees it: a list
 * of parts. The spec has no string form for a user turn, so such a middleware
 * does not check for one — which is exactly why handing it a string breaks it.
 */
const partsOf = (content: V3UserMessage["content"]): V3UserParts =>
  content as V3UserParts;

/**
 * A `transformParams` middleware that records the params it was given before
 * applying `rewrite`. `seen` is the precondition for every case that asserts
 * on a rewrite: an empty `seen` means the middleware never ran, and whatever
 * the wire then shows says nothing about it.
 */
function paramsProbe(
  id: string,
  rewrite: (params: MiddlewareParams) => MiddlewareParams,
): { middleware: MiddlewareFactoryOptions; seen: MiddlewareParams[] } {
  const seen: MiddlewareParams[] = [];
  const probe: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id, name: id },
    transformParams: async ({ params }) => {
      seen.push(params);
      return rewrite(params);
    },
  };
  return {
    middleware: { middleware: [probe], enabledMiddleware: [id] },
    seen,
  };
}

const promptRewrite =
  (rewrite: (prompt: V3Prompt) => V3Prompt) =>
  (params: MiddlewareParams): MiddlewareParams => ({
    ...params,
    prompt: rewrite(params.prompt),
  });

/** A middleware that observes and changes nothing, counting which hook ran. */
function countingProbe(id: string): {
  middleware: MiddlewareFactoryOptions;
  counts: { generate: number; stream: number };
} {
  const counts = { generate: 0, stream: 0 };
  const probe: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id, name: id },
    wrapGenerate: async ({ doGenerate }) => {
      counts.generate += 1;
      return doGenerate();
    },
    wrapStream: async ({ doStream }) => {
      counts.stream += 1;
      return doStream();
    },
  };
  return {
    middleware: { middleware: [probe], enabledMiddleware: [id] },
    counts,
  };
}

function captureLifecycle(): {
  finishes: LifecycleFinishPayload[];
  errors: LifecycleErrorPayload[];
  chunks: LifecycleChunkPayload[];
  onFinish: OnFinishCallback;
  onError: OnErrorCallback;
  onChunk: OnChunkCallback;
} {
  const finishes: LifecycleFinishPayload[] = [];
  const errors: LifecycleErrorPayload[] = [];
  const chunks: LifecycleChunkPayload[] = [];
  return {
    finishes,
    errors,
    chunks,
    onFinish: (payload) => {
      finishes.push(payload);
    },
    onError: (payload) => {
      errors.push(payload);
    },
    onChunk: (payload) => {
      chunks.push(payload);
    },
  };
}

// ---------------------------------------------------------------------------
// transformParams reaching the wire — one case per native loop.
// ---------------------------------------------------------------------------

await test("vertex Gemini native generate: a transformParams rewrite reaches the wire", async () => {
  const server = await startGeminiStandIn(() =>
    geminiTextTurn("gemini generate reply"),
  );
  const restoreEnv = withVertexEnv();
  const record = emptyRecord();
  const sdk = new NeuroLink();
  try {
    const result = await sdk.generate({
      input: { text: "hello vertex" },
      provider: "vertex",
      model: GEMINI_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    if (record.transformParamsCalls.length === 0) {
      throw new Error(
        "the probe never ran on the vertex Gemini native generate path",
      );
    }
    if (server.calls.length === 0) {
      throw new Error("the vertex Gemini stand-in was never reached");
    }
    const userText = geminiWire(server.calls[0]?.body ?? {}).userTexts.join(
      "\n",
    );
    if (!userText.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on vertex Gemini native generate",
      );
    }
    assert.ok(
      userText.includes("hello vertex"),
      "the caller's own text was dropped when the middleware appended a message",
    );
    assert.equal(
      typeof result.content,
      "string",
      "the native generate call did not return text content",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

await test("vertex Gemini native stream: a transformParams rewrite reaches the wire and wrapStream observes real usage", async () => {
  const server = await startGeminiStandIn(() =>
    geminiTextTurn("gemini stream reply"),
  );
  const restoreEnv = withVertexEnv();
  const record = emptyRecord();
  const capture: { finish: VertexCapturedFinish | undefined } = {
    finish: undefined,
  };
  const probe: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: { id: "vertex-stream-probe", name: "Vertex stream probe" },
    transformParams: async ({ type, params }) => {
      record.transformParamsCalls.push(type);
      return {
        ...params,
        prompt: [
          ...params.prompt,
          { role: "user", content: [{ type: "text", text: MARKER }] },
        ],
      };
    },
    wrapStream: async ({ doStream }) => {
      record.wrapStreamCalls += 1;
      const { stream } = await doStream();
      const reader = stream.getReader();
      const relay = new ReadableStream<LanguageModelV3StreamPart>({
        async pull(controller) {
          const next = await reader.read();
          if (next.done) {
            controller.close();
            return;
          }
          if (next.value.type === "finish") {
            capture.finish = {
              finishReasonUnified: next.value.finishReason.unified,
              inputTokens: next.value.usage.inputTokens.total,
              outputTokens: next.value.usage.outputTokens.total,
            };
          }
          controller.enqueue(next.value);
        },
        async cancel(reason) {
          await reader.cancel(reason);
        },
      });
      return { stream: relay };
    },
  };
  const sdk = new NeuroLink();
  try {
    const result = await sdk.stream({
      input: { text: "hello vertex" },
      provider: "vertex",
      model: GEMINI_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: {
        middleware: [probe],
        enabledMiddleware: ["vertex-stream-probe"],
      },
    });
    let text = "";
    for await (const chunk of result.stream) {
      if ("content" in chunk && typeof chunk.content === "string") {
        text += chunk.content;
      }
    }
    if (server.calls.length === 0) {
      throw new Error("the vertex Gemini stand-in was never reached");
    }
    const userText = geminiWire(server.calls[0]?.body ?? {}).userTexts.join(
      "\n",
    );
    if (!userText.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on vertex Gemini native stream",
      );
    }
    assert.ok(
      userText.includes("hello vertex"),
      "the caller's own text was dropped when the middleware appended a message",
    );
    assert.ok(
      text.includes("gemini stream reply"),
      "the native reply text did not reach the consumer",
    );
    if (!capture.finish) {
      throw new Error("wrapStream never observed a finish part");
    }
    assert.ok(
      typeof capture.finish.inputTokens === "number" &&
        capture.finish.inputTokens > 0,
      "wrapStream observed no real input token usage from the native call",
    );
    assert.ok(
      typeof capture.finish.outputTokens === "number" &&
        capture.finish.outputTokens > 0,
      "wrapStream observed no real output token usage from the native call",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

await test("vertex Anthropic native generate: a transformParams rewrite reaches the wire", async () => {
  const server = await startAnthropicStandIn(() =>
    anthropicTextTurn("anthropic generate reply"),
  );
  const restoreEnv = withVertexEnv();
  const record = emptyRecord();
  const sdk = new NeuroLink();
  try {
    const result = await sdk.generate({
      input: { text: "hello vertex claude" },
      provider: "vertex",
      model: ANTHROPIC_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    if (record.transformParamsCalls.length === 0) {
      throw new Error(
        "the probe never ran on the vertex Anthropic native generate path",
      );
    }
    if (server.calls.length === 0) {
      throw new Error("the vertex Anthropic stand-in was never reached");
    }
    const userText = anthropicWire(server.calls[0]?.body ?? {}).userTexts.join(
      "\n",
    );
    if (!userText.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on vertex Anthropic native generate",
      );
    }
    assert.ok(
      userText.includes("hello vertex claude"),
      "the caller's own text was dropped when the middleware appended a message",
    );
    assert.equal(
      typeof result.content,
      "string",
      "the native generate call did not return text content",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

await test("vertex Anthropic native stream: a transformParams rewrite reaches the wire", async () => {
  const server = await startAnthropicStandIn(() =>
    anthropicTextTurn("anthropic stream reply"),
  );
  const restoreEnv = withVertexEnv();
  const record = emptyRecord();
  const sdk = new NeuroLink();
  try {
    const result = await sdk.stream({
      input: { text: "hello vertex claude" },
      provider: "vertex",
      model: ANTHROPIC_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: middlewareOptions(record),
    });
    let text = "";
    for await (const chunk of result.stream) {
      if ("content" in chunk && typeof chunk.content === "string") {
        text += chunk.content;
      }
    }
    if (record.transformParamsCalls.length === 0) {
      throw new Error(
        "the probe never ran on the vertex Anthropic native stream path",
      );
    }
    if (server.calls.length === 0) {
      throw new Error("the vertex Anthropic stand-in was never reached");
    }
    const userText = anthropicWire(server.calls[0]?.body ?? {}).userTexts.join(
      "\n",
    );
    if (!userText.includes(MARKER)) {
      throw new Error(
        "the transformParams rewrite did not reach the wire on vertex Anthropic native stream",
      );
    }
    assert.ok(
      userText.includes("hello vertex claude"),
      "the caller's own text was dropped when the middleware appended a message",
    );
    assert.ok(
      text.includes("anthropic stream reply"),
      "the native reply text did not reach the consumer",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// wrapGenerate observing the native result (generate side of the pair above,
// which covered wrapStream).
// ---------------------------------------------------------------------------

await test("vertex Anthropic native generate: wrapGenerate observes a finishReason and real usage from the native call", async () => {
  const server = await startAnthropicStandIn(() =>
    anthropicTextTurn("anthropic generate reply"),
  );
  const restoreEnv = withVertexEnv();
  const capture: { finish: VertexCapturedFinish | undefined } = {
    finish: undefined,
  };
  const probe: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "vertex-generate-usage-probe",
      name: "Vertex generate usage probe",
    },
    wrapGenerate: async ({ doGenerate }) => {
      const generated = await doGenerate();
      capture.finish = {
        finishReasonUnified: generated.finishReason.unified,
        inputTokens: generated.usage.inputTokens.total,
        outputTokens: generated.usage.outputTokens.total,
      };
      return generated;
    },
  };
  const sdk = new NeuroLink();
  try {
    await sdk.generate({
      input: { text: "hello vertex claude" },
      provider: "vertex",
      model: ANTHROPIC_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: {
        middleware: [probe],
        enabledMiddleware: ["vertex-generate-usage-probe"],
      },
    });
    if (!capture.finish) {
      throw new Error("wrapGenerate never observed a result");
    }
    assert.equal(
      capture.finish.finishReasonUnified,
      "stop",
      "the native finish reason was not carried onto the V3 result",
    );
    assert.ok(
      typeof capture.finish.inputTokens === "number" &&
        capture.finish.inputTokens > 0,
      "wrapGenerate observed no real input token usage from the native call",
    );
    assert.ok(
      typeof capture.finish.outputTokens === "number" &&
        capture.finish.outputTokens > 0,
      "wrapGenerate observed no real output token usage from the native call",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// Precall guardrails blocking the call before it reaches the stand-in.
//
// The blocked-branch code in runNativeGenerateWithMiddleware /
// runNativeStreamWithMiddleware is provider-agnostic — it never inspects
// which native loop `callNative` closes over — so exercising it once per
// method (both on Gemini) proves the mechanism for all four loops; it is not
// Gemini-specific.
// ---------------------------------------------------------------------------

await test("vertex Gemini native generate: a precall guardrail blocks the call before it reaches the stand-in", async () => {
  const server = await startGeminiStandIn(() =>
    geminiTextTurn("should never be requested"),
  );
  const restoreEnv = withVertexEnv();
  let invoked = false;
  const blockingMiddleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "vertex-block-gemini-generate",
      name: "Block vertex Gemini generate",
    },
    wrapGenerate: async () => {
      invoked = true;
      return {
        content: [{ type: "text", text: "BLOCKED" }],
        finishReason: { unified: "stop" },
        usage: { inputTokens: { total: 0 }, outputTokens: { total: 0 } },
      };
    },
  };
  const sdk = new NeuroLink();
  try {
    const result = await sdk.generate({
      input: { text: "block this" },
      provider: "vertex",
      model: GEMINI_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: {
        middleware: [blockingMiddleware],
        enabledMiddleware: ["vertex-block-gemini-generate"],
      },
    });
    assert.ok(invoked, "blocking middleware did not run");
    assert.equal(
      server.calls.length,
      0,
      "a call a guardrail blocked still reached the vertex Gemini stand-in",
    );
    assert.equal(
      result.content,
      "BLOCKED",
      "the guardrail's synthesized content was lost",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

await test("vertex Gemini native stream: a precall guardrail blocks the call before it reaches the stand-in", async () => {
  const server = await startGeminiStandIn(() =>
    geminiTextTurn("should never be requested"),
  );
  const restoreEnv = withVertexEnv();
  let invoked = false;
  const blockingMiddleware: NeuroLinkMiddleware = {
    specificationVersion: "v3",
    metadata: {
      id: "vertex-block-gemini-stream",
      name: "Block vertex Gemini stream",
    },
    wrapStream: async () => {
      invoked = true;
      return {
        stream: new ReadableStream<LanguageModelV3StreamPart>({
          start(controller) {
            controller.enqueue({ type: "text-start", id: "blocked" });
            controller.enqueue({
              type: "text-delta",
              id: "blocked",
              delta: "BLOCKED",
            });
            controller.enqueue({ type: "text-end", id: "blocked" });
            controller.enqueue({
              type: "finish",
              finishReason: { unified: "stop" },
              usage: {
                inputTokens: { total: 0 },
                outputTokens: { total: 0 },
              },
            });
            controller.close();
          },
        }),
      };
    },
  };
  const sdk = new NeuroLink();
  try {
    const result = await sdk.stream({
      input: { text: "block this" },
      provider: "vertex",
      model: GEMINI_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: {
        middleware: [blockingMiddleware],
        enabledMiddleware: ["vertex-block-gemini-stream"],
      },
    });
    let text = "";
    for await (const chunk of result.stream) {
      if ("content" in chunk && typeof chunk.content === "string") {
        text += chunk.content;
      }
    }
    assert.equal(text, "BLOCKED", "blocked content lost");
    assert.ok(invoked, "blocking middleware did not run");
    assert.equal(
      server.calls.length,
      0,
      "a stream a guardrail blocked still reached the vertex Gemini stand-in",
    );
  } finally {
    await sdk.shutdown();
    restoreEnv();
    await server.close();
  }
});

// ---------------------------------------------------------------------------
// What a middleware edit does to the turn — every native loop.
//
// Each case reads the wire request through the loop's own reader instead of
// searching the serialized body. Every assertion about something a request
// LACKS follows the proof that the request existed, that the middleware ran,
// and — where a removal is asserted — that a control run carried the thing in
// the first place; a request that was never sent would otherwise "lack" all of
// it.
// ---------------------------------------------------------------------------

const EDITED_SYSTEM_TEXT = "EDITED_SYSTEM_RULES";
const ADDED_SYSTEM_TEXT = "ADDED_SYSTEM_RULES";
const APPENDED_TEXT = "APPENDED_CONTEXT";
const PREPENDED_TEXT = "PREPENDED_CONTEXT";
const SPREAD_APPENDED_TEXT = "SPREAD_APPENDED_CONTEXT";
const BLOCKED_TEXT = "BLOCKED";

const userTurn = (text: string): V3Prompt[number] => ({
  role: "user",
  content: [{ type: "text", text }],
});

const systemTurn = (text: string): V3Prompt[number] => ({
  role: "system",
  content: text,
});

const withoutSystem = (prompt: V3Prompt): V3Prompt =>
  prompt.filter((message) => message.role !== "system");

/** Applies `edit` to every text a user message carries, whatever its shape. */
const editUserText =
  (edit: (text: string) => string) =>
  (prompt: V3Prompt): V3Prompt =>
    prompt.map((message) => {
      if (message.role !== "user") {
        return message;
      }
      return typeof message.content === "string"
        ? { ...message, content: edit(message.content) }
        : {
            ...message,
            content: message.content.map((part) =>
              part.type === "text" ? { ...part, text: edit(part.text) } : part,
            ),
          };
    });

const joinedUserText = (wire: VertexWire): string => wire.userTexts.join("\n");

for (const loop of VERTEX_LOOPS) {
  await test(`vertex ${loop.label}: a middleware can edit, remove and add the system prompt`, async () => {
    const control = await runVertexLoop(loop, { systemPrompt: SYSTEM_TEXT });
    assert.equal(
      wireOf(loop, control).system,
      SYSTEM_TEXT,
      "the caller's system prompt did not reach the wire when no middleware touched it",
    );

    const edit = paramsProbe(
      "system-edit",
      promptRewrite((prompt) =>
        prompt.map((message) =>
          message.role === "system"
            ? { ...message, content: EDITED_SYSTEM_TEXT }
            : message,
        ),
      ),
    );
    const edited = await runVertexLoop(loop, {
      systemPrompt: SYSTEM_TEXT,
      middleware: edit.middleware,
    });
    assert.ok(
      edit.seen[0]?.prompt.some(
        (message) =>
          message.role === "system" && message.content === SYSTEM_TEXT,
      ),
      "the middleware was not shown the caller's system prompt",
    );
    assert.equal(
      wireOf(loop, edited).system,
      EDITED_SYSTEM_TEXT,
      "a system prompt the middleware edited did not reach the wire",
    );

    const removal = paramsProbe("system-remove", promptRewrite(withoutSystem));
    const removed = await runVertexLoop(loop, {
      systemPrompt: SYSTEM_TEXT,
      middleware: removal.middleware,
    });
    assert.ok(
      removal.seen.length > 0,
      "the system-removal middleware never ran",
    );
    const removedWire = wireOf(loop, removed);
    assert.ok(
      joinedUserText(removedWire).includes(USER_TEXT),
      "the user turn did not reach the wire in the removal run",
    );
    assert.equal(
      removedWire.system,
      undefined,
      "a system prompt the middleware removed still reached the wire",
    );

    const addition = paramsProbe(
      "system-add",
      promptRewrite((prompt) => [systemTurn(ADDED_SYSTEM_TEXT), ...prompt]),
    );
    const added = await runVertexLoop(loop, {
      middleware: addition.middleware,
    });
    assert.ok(
      addition.seen.length > 0,
      "the system-addition middleware never ran",
    );
    assert.equal(
      wireOf(loop, added).system,
      ADDED_SYSTEM_TEXT,
      "a system message the middleware added did not reach the wire",
    );
  });
}

for (const loop of VERTEX_LOOPS) {
  await test(`vertex ${loop.label}: a user message a middleware appends or prepends reaches the wire beside the caller's own`, async () => {
    const append = paramsProbe(
      "user-append",
      promptRewrite((prompt) => [...prompt, userTurn(APPENDED_TEXT)]),
    );
    const appended = await runVertexLoop(loop, {
      middleware: append.middleware,
    });
    assert.ok(append.seen.length > 0, "the append middleware never ran");
    assert.deepEqual(
      wireOf(loop, appended).userTexts,
      [`${USER_TEXT}\n\n${APPENDED_TEXT}`],
      "an appended user message did not reach the wire after the caller's text",
    );

    const prepend = paramsProbe(
      "user-prepend",
      promptRewrite((prompt) => [userTurn(PREPENDED_TEXT), ...prompt]),
    );
    const prepended = await runVertexLoop(loop, {
      middleware: prepend.middleware,
    });
    assert.ok(prepend.seen.length > 0, "the prepend middleware never ran");
    assert.deepEqual(
      wireOf(loop, prepended).userTexts,
      [`${PREPENDED_TEXT}\n\n${USER_TEXT}`],
      "a prepended user message did not reach the wire before the caller's text",
    );

    const edit = paramsProbe(
      "user-edit",
      promptRewrite(editUserText((text) => `${text} EDITED`)),
    );
    const edited = await runVertexLoop(loop, { middleware: edit.middleware });
    assert.ok(edit.seen.length > 0, "the edit middleware never ran");
    assert.deepEqual(
      wireOf(loop, edited).userTexts,
      [`${USER_TEXT} EDITED`],
      "an in-place edit of the user text did not reach the wire",
    );
  });
}

for (const loop of VERTEX_LOOPS) {
  await test(`vertex ${loop.label}: a middleware written against the V3 prompt shape sees the user turn as parts`, async () => {
    const probe = paramsProbe(
      "v3-parts",
      promptRewrite((prompt) =>
        prompt.map((message) =>
          message.role === "user" && Array.isArray(message.content)
            ? {
                ...message,
                content: [
                  ...message.content.map((part) =>
                    part.type === "text"
                      ? { ...part, text: part.text.toLowerCase() }
                      : part,
                  ),
                  { type: "text" as const, text: SPREAD_APPENDED_TEXT },
                ],
              }
            : message,
        ),
      ),
    );
    const run = await runVertexLoop(loop, { middleware: probe.middleware });
    assert.ok(probe.seen.length > 0, "the V3-shape middleware never ran");
    const shown = probe.seen[0]?.prompt.find(
      (message) => message.role === "user",
    );
    assert.ok(
      shown !== undefined && Array.isArray(shown.content),
      "the user turn reached the middleware as a bare string instead of V3 parts",
    );
    assert.deepEqual(
      wireOf(loop, run).userTexts,
      [`${USER_TEXT.toLowerCase()}\n\n${SPREAD_APPENDED_TEXT}`],
      "a rewrite written against V3 user parts did not reach the wire",
    );
  });
}

for (const loop of VERTEX_LOOPS) {
  await test(`vertex ${loop.label}: sampling params reach the wire, and a middleware can change or remove them`, async () => {
    const sampling = { temperature: 0.31, topP: 0.87, maxTokens: 32 };

    const inert = paramsProbe("sampling-inert", (params) => params);
    const control = await runVertexLoop(loop, {
      ...sampling,
      middleware: inert.middleware,
    });
    assert.equal(
      inert.seen[0]?.temperature,
      sampling.temperature,
      "the middleware was not shown the caller's temperature",
    );
    const controlWire = wireOf(loop, control);
    assert.deepEqual(
      [controlWire.temperature, controlWire.topP, controlWire.maxTokens],
      [sampling.temperature, sampling.topP, sampling.maxTokens],
      "the caller's sampling params did not reach the wire through an inert middleware",
    );

    const change = paramsProbe("sampling-edit", (params) => ({
      ...params,
      temperature: 0.55,
      topP: 0.66,
      maxOutputTokens: 48,
    }));
    const changed = await runVertexLoop(loop, {
      ...sampling,
      middleware: change.middleware,
    });
    assert.ok(change.seen.length > 0, "the sampling-edit middleware never ran");
    const changedWire = wireOf(loop, changed);
    assert.deepEqual(
      [changedWire.temperature, changedWire.topP, changedWire.maxTokens],
      [0.55, 0.66, 48],
      "sampling params a middleware changed did not reach the wire",
    );

    const removal = paramsProbe("sampling-remove", (params) => ({
      ...params,
      temperature: undefined,
      topP: undefined,
      maxOutputTokens: undefined,
    }));
    const removed = await runVertexLoop(loop, {
      ...sampling,
      middleware: removal.middleware,
    });
    assert.equal(
      removal.seen[0]?.topP,
      sampling.topP,
      "the middleware was not shown the topP it then removed",
    );
    const removedWire = wireOf(loop, removed);
    assert.ok(
      joinedUserText(removedWire).includes(USER_TEXT),
      "the user turn did not reach the wire in the removal run",
    );
    assert.notEqual(
      removedWire.temperature,
      sampling.temperature,
      "a temperature the middleware removed still reached the wire",
    );
    assert.equal(
      removedWire.topP,
      undefined,
      "a topP the middleware removed still reached the wire",
    );
    assert.notEqual(
      removedWire.maxTokens,
      sampling.maxTokens,
      "a token cap the middleware removed still reached the wire",
    );
  });
}

// ---------------------------------------------------------------------------
// Prompt items a native request has no place for.
// ---------------------------------------------------------------------------

const UNMAPPED_FILE_PAYLOAD = "UNMAPPED_FILE_PAYLOAD";
const UNMAPPED_ASSISTANT_TEXT = "UNMAPPED_ASSISTANT_TEXT";

/** Runs `body` while recording the first argument of every `logger.warn`. */
async function captureWarnings<T>(
  body: () => Promise<T>,
): Promise<{ result: T; warnings: string[] }> {
  const warnings: string[] = [];
  const original = logger.warn;
  logger.warn = (...args: unknown[]) => {
    warnings.push(String(args[0]));
  };
  try {
    return { result: await body(), warnings };
  } finally {
    logger.warn = original;
  }
}

const isUnmappedReport = (warning: string): boolean =>
  /prompt item/.test(warning);

for (const loop of VERTEX_LOOPS) {
  await test(`vertex ${loop.label}: prompt items the native request cannot carry are reported, not silently dropped`, async () => {
    const inert = paramsProbe("unmapped-control", (params) => params);
    const control = await captureWarnings(() =>
      runVertexLoop(loop, { middleware: inert.middleware }),
    );
    assert.ok(inert.seen.length > 0, "the control middleware never ran");
    assert.ok(
      joinedUserText(wireOf(loop, control.result)).includes(USER_TEXT),
      "the control request never carried the user turn",
    );
    assert.equal(
      control.warnings.filter(isUnmappedReport).length,
      0,
      "a middleware that added nothing was warned about",
    );

    const adding = paramsProbe(
      "unmapped-add",
      promptRewrite((prompt) => [
        ...prompt,
        {
          role: "user",
          content: [
            {
              type: "file",
              data: UNMAPPED_FILE_PAYLOAD,
              mediaType: "text/plain",
            },
          ],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: UNMAPPED_ASSISTANT_TEXT }],
        },
      ]),
    );
    const { result: run, warnings } = await captureWarnings(() =>
      runVertexLoop(loop, { middleware: adding.middleware }),
    );
    assert.ok(adding.seen.length > 0, "the item-adding middleware never ran");
    assert.deepEqual(
      wireOf(loop, run).userTexts,
      [USER_TEXT],
      "the user turn changed although the middleware added no text",
    );
    const reports = warnings.filter(isUnmappedReport);
    assert.equal(
      reports.length,
      1,
      "the dropped prompt items were not reported exactly once",
    );
    assert.match(
      reports[0] ?? "",
      /2 prompt item/,
      "the report did not count both dropped items",
    );
    const everywhere = `${warnings.join("\n")}\n${JSON.stringify(run.bodies[0])}`;
    assert.ok(
      !everywhere.includes(UNMAPPED_FILE_PAYLOAD) &&
        !everywhere.includes(UNMAPPED_ASSISTANT_TEXT),
      "an item the request cannot carry leaked into the wire or the warning",
    );
  });
}

// ---------------------------------------------------------------------------
// Lifecycle callbacks — fired once, from the right place.
//
// Three things can fire a caller's onFinish/onError/onChunk on a native Vertex
// call: BaseProvider's stream wrapper, Vertex's own manual firing, and the
// built-in lifecycle middleware inside the model bridge. The bridge one reads
// legacy field names a V3 model never produces, so it reports a text of
// "undefined" and a finishReason of "[object Object]". Every count here sits
// next to a floor that proves the callbacks were live, since a ceiling alone
// also passes for callbacks that never fire.
// ---------------------------------------------------------------------------

const withDirectLifecycle = (
  middleware: MiddlewareFactoryOptions,
  lifecycle: ReturnType<typeof captureLifecycle>,
): MiddlewareFactoryOptions => ({
  ...middleware,
  middlewareConfig: {
    lifecycle: {
      enabled: true,
      config: {
        onFinish: lifecycle.onFinish,
        onError: lifecycle.onError,
        onChunk: lifecycle.onChunk,
      },
    },
  },
});

for (const loop of GENERATE_LOOPS) {
  await test(`vertex ${loop.label}: top-level onFinish and onError fire once with middleware configured`, async () => {
    const observed = countingProbe("lifecycle-generate-probe");
    const lifecycle = captureLifecycle();
    const run = await runVertexLoop(loop, {
      middleware: observed.middleware,
      onFinish: lifecycle.onFinish,
      onError: lifecycle.onError,
    });
    assert.equal(
      run.text,
      VERTEX_REPLY,
      "the native turn did not return the stand-in's reply",
    );
    assert.ok(
      observed.counts.generate > 0,
      "the middleware chain never wrapped the generate call",
    );
    assert.ok(
      lifecycle.finishes.length >= 1,
      "onFinish never fired, so its count says nothing",
    );
    assert.equal(
      lifecycle.finishes.length,
      1,
      "onFinish fired more than once for one generate call",
    );
    assert.equal(
      lifecycle.finishes[0]?.finishReason,
      "stop",
      "onFinish did not carry the turn's real finish reason",
    );
    assert.equal(
      lifecycle.errors.length,
      0,
      "onError fired for a generate call that succeeded",
    );
  });

  await test(`vertex ${loop.label}: a lifecycle middleware configured directly still fires once`, async () => {
    const observed = countingProbe("lifecycle-direct-generate-probe");
    const lifecycle = captureLifecycle();
    const run = await runVertexLoop(loop, {
      middleware: withDirectLifecycle(observed.middleware, lifecycle),
    });
    assert.equal(
      run.text,
      VERTEX_REPLY,
      "the native turn did not return the stand-in's reply",
    );
    assert.ok(
      observed.counts.generate > 0,
      "the middleware chain never wrapped the generate call",
    );
    assert.equal(
      lifecycle.finishes.length,
      1,
      "a directly configured lifecycle middleware did not fire onFinish exactly once",
    );
    assert.equal(
      lifecycle.errors.length,
      0,
      "onError fired for a generate call that succeeded",
    );
  });
}

for (const loop of STREAM_LOOPS) {
  await test(`vertex ${loop.label}: top-level callbacks are not fired a third time by the bridge`, async () => {
    const observed = countingProbe("lifecycle-stream-probe");
    const lifecycle = captureLifecycle();
    const run = await runVertexLoop(loop, {
      middleware: observed.middleware,
      onFinish: lifecycle.onFinish,
      onError: lifecycle.onError,
      onChunk: lifecycle.onChunk,
    });
    assert.equal(
      run.text,
      VERTEX_REPLY,
      "the native turn did not return the stand-in's reply",
    );
    assert.ok(
      observed.counts.stream > 0,
      "the middleware chain never wrapped the stream call",
    );
    assert.ok(
      lifecycle.finishes.length >= 1,
      "onFinish never fired, so its count says nothing",
    );
    assert.ok(
      lifecycle.chunks.length >= 1,
      "onChunk never fired, so its count says nothing",
    );
    // BaseProvider's wrapper and Vertex's own firing already make a pair on a
    // stream; the bridge's middleware must not add a third.
    assert.ok(
      lifecycle.finishes.length <= 2,
      "onFinish fired more than the provider's own two paths can",
    );
    assert.ok(
      lifecycle.chunks.length <= 2,
      "onChunk fired more than the provider's own two paths can",
    );
    assert.ok(
      lifecycle.finishes.every((payload) => payload.text === VERTEX_REPLY),
      "an onFinish payload did not carry the streamed text",
    );
    assert.ok(
      lifecycle.chunks.every(
        (payload) =>
          payload.type !== "text-delta" || payload.textDelta === VERTEX_REPLY,
      ),
      "an onChunk payload did not carry the streamed delta",
    );
    assert.equal(
      lifecycle.errors.length,
      0,
      "onError fired for a stream that succeeded",
    );
  });

  await test(`vertex ${loop.label}: a lifecycle middleware configured directly fires each callback once`, async () => {
    const observed = countingProbe("lifecycle-direct-stream-probe");
    const lifecycle = captureLifecycle();
    const run = await runVertexLoop(loop, {
      middleware: withDirectLifecycle(observed.middleware, lifecycle),
    });
    assert.equal(
      run.text,
      VERTEX_REPLY,
      "the native turn did not return the stand-in's reply",
    );
    assert.ok(
      observed.counts.stream > 0,
      "the middleware chain never wrapped the stream call",
    );
    assert.ok(
      lifecycle.finishes.length >= 1,
      "onFinish never fired, so its count says nothing",
    );
    assert.equal(
      lifecycle.finishes.length,
      1,
      "a directly configured lifecycle fired onFinish more than once",
    );
    const deltas = lifecycle.chunks.filter(
      (payload) => payload.type === "text-delta",
    );
    assert.ok(
      deltas.length >= 1,
      "onChunk never saw a text delta, so its count says nothing",
    );
    assert.equal(
      deltas.length,
      1,
      "a directly configured lifecycle delivered the text delta more than once",
    );
    assert.ok(
      lifecycle.finishes.every((payload) => payload.text === VERTEX_REPLY) &&
        deltas.every((payload) => payload.textDelta === VERTEX_REPLY),
      "a lifecycle payload did not carry the streamed text",
    );
    assert.equal(
      lifecycle.errors.length,
      0,
      "onError fired for a stream that succeeded",
    );
  });
}

// ---------------------------------------------------------------------------
// Schema-complexity recovery is the provider's own business.
//
// Vertex answers an over-constrained responseSchema with a deterministic 400,
// and the Gemini generate loop retries once without the schema. Middleware
// wraps one model call, and the provider hands it one that succeeds.
// ---------------------------------------------------------------------------

const TOO_MANY_STATES: GeminiStandInReply = {
  status: 400,
  json: {
    error: {
      code: 400,
      message:
        "The specified schema produces a constraint that has too many states for serving. Simplify the schema.",
      status: "INVALID_ARGUMENT",
    },
  },
};

const carriesResponseSchema = (body: Record<string, unknown> | undefined) =>
  /responseSchema|responseJsonSchema/.test(JSON.stringify(body ?? {}));

for (const variant of ["top-level", "direct"] as const) {
  await test(`vertex Gemini native generate: a schema-complexity retry is one call to middleware and lifecycle callbacks (${variant} callbacks)`, async () => {
    const server = await startGeminiStandIn((index) =>
      index === 0 ? TOO_MANY_STATES : geminiTextTurn('{"answer":"retry ok"}'),
    );
    const restoreEnv = withVertexEnv();
    const observed = countingProbe("schema-retry-probe");
    const lifecycle = captureLifecycle();
    const schema = z.object({ answer: z.string() });
    const sdk = new NeuroLink();
    try {
      const result = await sdk.generate({
        input: { text: USER_TEXT },
        provider: "vertex",
        model: GEMINI_MODEL,
        maxTokens: 64,
        disableTools: true,
        disableInternalFallback: true,
        credentials: vertexCredentialsFor(server.port),
        schema,
        output: { format: "json" },
        middleware:
          variant === "direct"
            ? withDirectLifecycle(observed.middleware, lifecycle)
            : observed.middleware,
        ...(variant === "top-level"
          ? { onFinish: lifecycle.onFinish, onError: lifecycle.onError }
          : {}),
      });
      await settleLifecycle();
      assert.equal(
        server.calls.length,
        2,
        "the schema-complexity error did not produce exactly one retry",
      );
      assert.ok(
        carriesResponseSchema(server.calls[0]?.body),
        "the first request carried no response schema, so the retry has nothing to drop",
      );
      assert.ok(
        !carriesResponseSchema(server.calls[1]?.body),
        "the retry still carried the schema the provider rejected",
      );
      assert.equal(
        asRecord(asRecord(server.calls[1]?.body).generationConfig)
          .responseMimeType,
        "application/json",
        "the retry lost JSON mode along with the schema",
      );
      assert.equal(
        schema.parse(result.structuredData).answer,
        "retry ok",
        "the retry's JSON did not reach the caller as structured data",
      );
      assert.ok(
        lifecycle.finishes.length >= 1,
        "onFinish never fired, so its count says nothing",
      );
      assert.equal(
        observed.counts.generate,
        1,
        "middleware saw the provider-internal retry as a second generate call",
      );
      assert.equal(
        lifecycle.finishes.length,
        1,
        "onFinish did not fire exactly once for the recovered call",
      );
      assert.equal(
        lifecycle.errors.length,
        0,
        "onError fired for a call that recovered",
      );
    } finally {
      await sdk.shutdown();
      restoreEnv();
      await server.close();
    }
  });
}

// ---------------------------------------------------------------------------
// A guardrail that blocks generate() before the native call.
// ---------------------------------------------------------------------------

for (const loop of GENERATE_LOOPS) {
  await test(`vertex ${loop.label}: a guardrail-blocked call still returns analytics`, async () => {
    const control = await runVertexLoop(loop, { enableAnalytics: true });
    assert.ok(
      control.analytics !== undefined,
      "an unblocked call returned no analytics, so the blocked comparison has no baseline",
    );
    assert.equal(
      control.analytics.stepsUsed,
      1,
      "the unblocked control did not report its one step",
    );
    assert.ok(
      control.analytics.tokenUsage.total > 0,
      "the unblocked control reported no token usage",
    );

    let blockedCalls = 0;
    const blocker: NeuroLinkMiddleware = {
      specificationVersion: "v3",
      metadata: { id: "block-for-analytics", name: "Block for analytics" },
      wrapGenerate: async () => {
        blockedCalls += 1;
        return {
          content: [{ type: "text", text: BLOCKED_TEXT }],
          finishReason: { unified: "stop" },
          usage: { inputTokens: { total: 0 }, outputTokens: { total: 0 } },
        };
      },
    };
    const blocked = await runVertexLoop(loop, {
      enableAnalytics: true,
      middleware: {
        middleware: [blocker],
        enabledMiddleware: ["block-for-analytics"],
      },
    });
    assert.equal(
      blockedCalls,
      1,
      "the blocking middleware did not run exactly once",
    );
    assert.equal(
      blocked.text,
      BLOCKED_TEXT,
      "the guardrail's synthesized content was lost",
    );
    assert.equal(
      blocked.bodies.length,
      0,
      "a call the guardrail blocked still reached the stand-in",
    );
    const analytics = blocked.analytics;
    assert.ok(
      analytics !== undefined,
      "a blocked call returned no analytics although enableAnalytics was set",
    );
    assert.equal(
      analytics.stepsUsed,
      0,
      "a blocked call reported a model step it never took",
    );
    assert.equal(
      analytics.toolCallCount,
      0,
      "a blocked call reported tool calls it never made",
    );
    assert.equal(
      analytics.tokenUsage.total,
      0,
      "a blocked call reported tokens it never used",
    );
  });
}

// ---------------------------------------------------------------------------
// Cancellation — a consumer that walks away from a stream must release the
// upstream request, not just stop reading it.
//
// The stand-in opens the response with the first frames of a turn and then
// holds it open, so nothing but the client aborting can end the connection.
// The bridge's ReadableStream is pulled ahead of demand, which leaves a
// `.next()` parked on the native channel when the consumer breaks. Releasing
// the native iterator by awaiting its `.return()` queues behind that
// `.next()`, and the break then stays parked until a frame or the turn clock
// arrives — with the upstream request still open and nothing left to end it.
//
// Only the Anthropic-on-Vertex loop can show this. executeNativeGemini3Stream
// collects the whole turn before anything reaches the bridge (see the
// "Collected, NOT forwarded to the consumer" comment at its `pump`), so by the
// time a consumer can break there is no request left to release.
// ---------------------------------------------------------------------------

const waitFor = async (
  condition: () => boolean,
  deadlineMs: number,
): Promise<boolean> => {
  const deadline = Date.now() + deadlineMs;
  while (!condition() && Date.now() < deadline) {
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
  }
  return condition();
};

await test("vertex Anthropic native stream: breaking out of the consumer's loop releases the upstream request", async () => {
  const server = await startAnthropicHoldOpenStandIn();
  const restoreEnv = withVertexEnv();
  const sdk = new NeuroLink();
  try {
    const result = await sdk.stream({
      input: { text: "hello vertex claude" },
      provider: "vertex",
      model: ANTHROPIC_MODEL,
      maxTokens: 32,
      disableTools: true,
      disableInternalFallback: true,
      credentials: vertexCredentialsFor(server.port),
      middleware: middlewareOptions(emptyRecord()),
    });
    let firstChunkSeen = false;
    const consumer = (async () => {
      for await (const chunk of result.stream) {
        if ("content" in chunk && chunk.content) {
          firstChunkSeen = true;
          break;
        }
      }
    })();
    const returned = await bounded(consumer, 10_000).then(
      () => true,
      () => false,
    );
    assert.ok(
      firstChunkSeen,
      "no content reached the consumer before it broke out of the loop",
    );
    assert.ok(
      returned,
      "the consumer's break did not return while the upstream response was held open",
    );
    assert.equal(
      server.calls.length,
      1,
      "the hold-open stand-in did not receive exactly one request",
    );
    assert.ok(
      await waitFor(() => server.closedResponses() >= 1, 10_000),
      "the upstream request stayed open after the consumer broke out of the stream",
    );
  } finally {
    await bounded(server.close());
    await bounded(sdk.shutdown());
    restoreEnv();
  }
});

await runSuite();
