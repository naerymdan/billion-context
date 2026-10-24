// #2666: a summary target that thinks by default spent its whole output budget
// on reasoning (finish_reason "length", empty content), so every fold failed and
// nothing in config could turn it off. The per-model recipe knob `reasoning`
// ("off" | "template-off") writes the protocol's thinking control into the
// summary request; without the knob the request body is unchanged.
import assert from "node:assert/strict";
import http from "node:http";
import { test } from "node:test";
import { ExternalSummaryExecutor, type SummaryWork } from "../src/external-summary.ts";
import { applySummaryReasoning, createSummaryHttpCandidate, type SummaryHttpTarget } from "../src/external-summary-http.ts";
import { expandExternalSummaryChain, parseExternalSummarySettings, parseSummaryReasoning, type SummaryReasoning } from "../src/external-summary-settings.ts";
import { parseNamedProviderRecipe } from "../src/config.ts";
import { parseAgentProviderReport } from "../src/agent-providers.ts";
import type { PreflightProtocol } from "../src/preflight.ts";

process.env.NODE_ENV = "test";
const work: SummaryWork = { content: "Prior evidence and decisions", reference: "Read-only current task", instructions: "Preserve unfinished objectives" };
const budget = { totalTimeoutMs: 4000, targetTimeoutMs: 1500, maxSummaryBytes: 2048 };
const summary = "Retained decisions, evidence, constraints and unfinished objectives.";
const THINKING_FIELDS = ["reasoning_effort", "reasoning", "thinking", "chat_template_kwargs"];

function completion(protocol: PreflightProtocol): object {
    if (protocol === "responses") return { status: "completed", output_text: summary };
    if (protocol === "anthropic") return { stop_reason: "end_turn", content: [{ type: "text", text: summary }] };
    if (protocol === "openai") return { choices: [{ message: { content: summary }, finish_reason: "stop" }] };
    return { candidates: [{ content: { parts: [{ text: summary }] }, finishReason: "STOP" }] };
}

/** Send one summary request through the real candidate and return the body the upstream received. */
async function capturedBody(protocol: PreflightProtocol, reasoning?: SummaryReasoning): Promise<Record<string, unknown>> {
    let body: Record<string, unknown> | undefined;
    const server = http.createServer((req, res) => {
        void (async () => {
            const chunks: Buffer[] = [];
            for await (const chunk of req) chunks.push(Buffer.from(chunk));
            body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
            res.setHeader("content-type", "application/json");
            res.end(JSON.stringify(completion(protocol)));
        })().catch(() => { res.writeHead(500); res.end(); });
    });
    await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    try {
        const target: SummaryHttpTarget = {
            url: `http://127.0.0.1:${address.port}`, protocol, model: "external-test-model",
            headers: { authorization: "Bearer test-summary-key" }, outputTokens: 1024, ...(reasoning !== undefined ? { reasoning } : {}),
        };
        const result = await new ExternalSummaryExecutor(1).execute(work, [createSummaryHttpCandidate(target, 4096)], budget);
        assert.equal(result.status, "success");
    } finally {
        await new Promise<void>((resolve) => { server.close(() => resolve()); server.closeAllConnections(); });
    }
    assert.ok(body, "upstream received a request");
    return body;
}

const OFF_EXPECTED: Record<PreflightProtocol, (body: Record<string, unknown>) => void> = {
    openai: (body) => assert.equal(body.reasoning_effort, "none"),
    responses: (body) => assert.deepEqual(body.reasoning, { effort: "none" }),
    anthropic: (body) => assert.deepEqual(body.thinking, { type: "disabled" }),
    google: (body) => {
        const generationConfig = body.generationConfig as Record<string, unknown>;
        assert.deepEqual(generationConfig.thinkingConfig, { thinkingBudget: 0 });
        assert.equal(generationConfig.maxOutputTokens, 1024, "the output budget survives next to the thinking control");
    },
};

for (const protocol of ["openai", "responses", "anthropic", "google"] as const) {
    test(`#2666: ${protocol} — reasoning "off" sends the protocol's own thinking control`, async () => {
        OFF_EXPECTED[protocol](await capturedBody(protocol, "off"));
    });

    test(`#2666: ${protocol} — no knob, no thinking field (body unchanged)`, async () => {
        const body = await capturedBody(protocol);
        for (const field of THINKING_FIELDS) assert.equal(body[field], undefined, `unconfigured ${protocol} body must not carry ${field}`);
        if (protocol === "google") assert.equal((body.generationConfig as Record<string, unknown>).thinkingConfig, undefined);
    });
}

test("#2666: openai — reasoning \"template-off\" sends chat_template_kwargs, not reasoning_effort", async () => {
    const body = await capturedBody("openai", "template-off");
    assert.deepEqual(body.chat_template_kwargs, { enable_thinking: false });
    assert.equal(body.reasoning_effort, undefined);
});

test("#2666: applySummaryReasoning keeps existing chat_template_kwargs and is a no-op without the knob", () => {
    const payload: Record<string, unknown> = { model: "m", chat_template_kwargs: { custom: 1 } };
    applySummaryReasoning(payload, "openai", "template-off");
    assert.deepEqual(payload.chat_template_kwargs, { custom: 1, enable_thinking: false });
    const untouched: Record<string, unknown> = { model: "m", max_tokens: 10 };
    applySummaryReasoning(untouched, "openai", undefined);
    assert.deepEqual(untouched, { model: "m", max_tokens: 10 });
});

test("#2666: parseSummaryReasoning accepts the two values and rejects the rest", () => {
    assert.equal(parseSummaryReasoning(undefined, "openai", "t"), undefined);
    assert.equal(parseSummaryReasoning("off", "anthropic", "t"), "off");
    assert.equal(parseSummaryReasoning("template-off", "openai", "t"), "template-off");
    assert.throws(() => parseSummaryReasoning("none", "openai", "t"), /reasoning must be "off" or "template-off"/);
    assert.throws(() => parseSummaryReasoning(false, "openai", "t"), /reasoning must be/);
    for (const protocol of ["anthropic", "responses", "google"]) {
        assert.throws(() => parseSummaryReasoning("template-off", protocol, "t"), /needs api "openai"/);
    }
});

const recipeInput = (models: Record<string, unknown>, api = "openai") => ({ baseUrl: "https://summary.example/v1", api, apiKeyEnv: "SUMMARY_KEY", models });

test("#2666: named recipe — reasoning reaches the expanded chain target", () => {
    const recipe = parseNamedProviderRecipe(recipeInput({ thinker: { outputTokens: 2048, reasoning: "template-off" }, plain: { outputTokens: 2048 } }));
    assert.equal(recipe.models.thinker.reasoning, "template-off");
    assert.equal("reasoning" in recipe.models.plain, false);
    const expanded = expandExternalSummaryChain({ enabled: true, targets: ["local/thinker", "local/plain"] }, { local: recipe });
    assert.equal(expanded.targets[0].reasoning, "template-off");
    assert.equal("reasoning" in expanded.targets[1], false, "an unconfigured model stays unconfigured");
});

test("#2666: named recipe — invalid reasoning values are rejected", () => {
    assert.throws(() => parseNamedProviderRecipe(recipeInput({ m: { reasoning: "minimal" } })), /reasoning must be/);
    assert.throws(() => parseNamedProviderRecipe(recipeInput({ m: { reasoning: "template-off" } }, "anthropic")), /needs api "openai"/);
    assert.throws(() => parseNamedProviderRecipe(recipeInput({ m: { thinking: "off" } })), /unknown fields on recipe model/, "other keys stay unknown");
});

test("#2666: request-rail settings accept reasoning and validate it against the protocol", () => {
    const base = { name: "local", protocol: "openai", url: "https://summary.example/v1/chat/completions", model: "m", credentialRef: "env:SUMMARY_KEY" };
    const parsed = parseExternalSummarySettings({ enabled: true, targets: [{ ...base, reasoning: "template-off" }] });
    assert.equal(parsed.targets[0].reasoning, "template-off");
    assert.equal("reasoning" in parseExternalSummarySettings({ enabled: true, targets: [base] }).targets[0], false);
    assert.throws(() => parseExternalSummarySettings({ enabled: true, targets: [{ ...base, protocol: "anthropic", url: "https://summary.example/v1/messages", reasoning: "template-off" }] }), /needs api "openai"/);
});

test("#2666: agent-registry recipes carry reasoning too; a bad value skips that provider", () => {
    const report = parseAgentProviderReport({ agent: "pi", providers: {
        good: { baseUrl: "https://summary.example/v1", api: "openai", apiKey: "k", models: { m: { reasoning: "off" } } },
        bad: { baseUrl: "https://summary.example/v1", api: "google", apiKey: "k", models: { m: { reasoning: "template-off" } } },
    } });
    assert.equal(report.registered.good.models.m.reasoning, "off");
    assert.equal(report.registered.bad, undefined);
    assert.equal(report.skipped[0]?.name, "bad");
});
