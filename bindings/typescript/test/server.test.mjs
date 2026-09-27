import assert from "node:assert/strict";
import { createServer } from "node:net";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { OpenKindServer, ServerError } from "../dist/server.js";

const fixture = fileURLToPath(new URL("../../test/fake_openkindd.py", import.meta.url));

async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const port = probe.address().port;
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

test("server owns its process and returns a working authenticated client", async () => {
  const server = new OpenKindServer({ binary: fixture, httpAddr: `127.0.0.1:${await freePort()}`,
    models: ["mock"], apiKey: "secret" });
  await server.start();
  try {
    assert.equal(server.running, true);
    assert.equal((await server.client().health()).data.status, "ok");
    assert.equal((await server.client().listModels()).data.models[0].name, "mock");
    const result = await server.client().systemOne("billing ticket", {
      billing: { type: "noul", instructions: "Is this billing?" },
      team: { type: "choice", instructions: "Which team?", criteria: { billing: null, sales: null } },
      severity: { type: "score", instructions: "Rate severity", criteria: ["low", "high"] },
    }, "mock");
    assert.deepEqual(Object.keys(result.data.answers).sort(), ["billing", "severity", "team"]);
    assert.equal(result.requestId, "fixture-request");
    assert.equal((await fetch(`${server.baseUrl}/v1/models`)).status, 401);
  } finally {
    await server.stop();
  }
  assert.equal(server.running, false);
});

test("server rejects occupied address without stopping its owner", async () => {
  const occupied = createServer();
  await new Promise((resolve) => occupied.listen(0, "127.0.0.1", resolve));
  try {
    const server = new OpenKindServer({ binary: fixture, httpAddr: `127.0.0.1:${occupied.address().port}` });
    await assert.rejects(server.start(), ServerError);
    assert.equal(server.running, false);
    assert.equal(occupied.listening, true);
  } finally {
    await new Promise((resolve) => occupied.close(resolve));
  }
});

test("server reports a missing executable", async () => {
  const server = new OpenKindServer({ binary: "/nonexistent/openkindd", httpAddr: `127.0.0.1:${await freePort()}` });
  await assert.rejects(server.start(), ServerError);
  assert.equal(server.running, false);
});

test("server wrapper starts a real openkindd binary", {
  skip: !process.env.OPENKIND_TEST_BINARY,
}, async () => {
  const server = new OpenKindServer({ binary: process.env.OPENKIND_TEST_BINARY,
    httpAddr: `127.0.0.1:${await freePort()}`, models: ["mock"], apiKey: "dev-key" });
  await server.start();
  try {
    assert.equal((await server.client().health()).data.status, "ok");
    const result = await server.client().systemOne("billing ticket", {
      billing: { type: "noul", instructions: "Is this billing?" },
    }, "mock");
    assert.equal(result.data.answers.billing.type, "noul");
  } finally {
    await server.stop();
  }
  assert.equal(server.running, false);
});
