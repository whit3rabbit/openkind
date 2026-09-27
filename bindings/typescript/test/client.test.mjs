import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, InvalidResponseError, OpenKindClient } from "../dist/index.js";

const question = { type: "choice", instructions: "Which team?", criteria: { billing: null, sales: "Sales" } };

test("evaluation sends the Jev body and checks the returned choice", async () => {
  let seen;
  const fetcher = async (url, options) => {
    seen = { url, options };
    return new Response(JSON.stringify({
      model: "mock",
      answers: { team: { type: "choice", choice: "billing", probabilities: { billing: 0.75, sales: 0.25 }, confidence: 0.75 } },
      usage: { input_tokens: 2, output_tokens: 1 },
    }), { status: 200, headers: { "x-typesafe-request-id": "request-123" } });
  };
  const client = new OpenKindClient({ baseUrl: "http://127.0.0.1:18080/", apiKey: "test-key", fetch: fetcher });
  const result = await client.systemOne("ticket", { team: question }, "mock");
  assert.equal(seen.url, "http://127.0.0.1:18080/v1/systemone");
  assert.equal(seen.options.headers.Authorization, "Bearer test-key");
  assert.deepEqual(JSON.parse(seen.options.body), { state: "ticket", model: "mock", questions: { team: question } });
  assert.equal(result.data.answers.team.choice, "billing");
  assert.equal(result.requestId, "request-123");
});

test("response validation rejects out-of-set decisions", async () => {
  const fetcher = async () => new Response(JSON.stringify({
    model: "mock",
    answers: { team: { type: "choice", choice: "outside", probabilities: { billing: 0.75, sales: 0.25 }, confidence: 0.75 } },
    usage: { input_tokens: 2, output_tokens: 1 },
  }), { status: 200 });
  const client = new OpenKindClient({ fetch: fetcher });
  await assert.rejects(client.systemOne("ticket", { team: question }), InvalidResponseError);
});

test("errors expose status, code, and request ID", async () => {
  const fetcher = async () => new Response(JSON.stringify({ error: { code: "rate_limited", message: "slow down" } }), {
    status: 429, headers: { "x-typesafe-request-id": "request-429" },
  });
  const client = new OpenKindClient({ fetch: fetcher });
  await assert.rejects(client.systemOne("ticket", { team: question }), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 429);
    assert.equal(error.code, "rate_limited");
    assert.equal(error.requestId, "request-429");
    return true;
  });
});

test("models and health use their own routes, without health auth", async () => {
  const seen = [];
  const fetcher = async (url, options) => {
    seen.push({ url, options });
    return new Response(url.endsWith("/health") ? '{"status":"ok"}' :
      '{"models":[{"name":"mock","description":"Mock engine","release_date":"2026-01-01"}]}', { status: 200 });
  };
  const client = new OpenKindClient({ apiKey: "test-key", fetch: fetcher });
  assert.equal((await client.listModels()).data.models[0].name, "mock");
  assert.equal((await client.health()).data.status, "ok");
  assert.equal(seen[0].options.headers.Authorization, "Bearer test-key");
  assert.equal(seen[1].options.headers.Authorization, undefined);
});

test("live daemon covers Noul, Choice, Score, models, and health", {
  skip: !process.env.OPENKIND_TEST_URL,
}, async () => {
  const client = new OpenKindClient({ baseUrl: process.env.OPENKIND_TEST_URL,
    apiKey: process.env.OPENKIND_TEST_API_KEY });
  const result = await client.systemOne("A customer was charged twice.", {
    billing: { type: "noul", instructions: "Is this a billing issue?" },
    team: { type: "choice", instructions: "Which team?", criteria: { billing: null, sales: null } },
    severity: { type: "score", instructions: "Rate severity", criteria: ["low", "high"] },
  }, "mock");
  assert.equal(result.data.model, "mock");
  assert.ok(result.requestId);
  assert.deepEqual(Object.keys(result.data.answers).sort(), ["billing", "severity", "team"]);
  assert.ok((await client.listModels()).data.models.some((model) => model.name === "mock"));
  assert.equal((await client.health()).data.status, "ok");
});
