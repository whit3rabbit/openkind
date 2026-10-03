import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { generateApiKey, OpenKindServer, ServerError } from "../dist/server.js";

const fixture = fileURLToPath(new URL("../../test/fake_openkindd.py", import.meta.url));

async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const port = probe.address().port;
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

test("API key generation returns fresh 256-bit keys", () => {
  const first = generateApiKey();
  const second = generateApiKey();
  assert.match(first, /^ok_[0-9a-f]{64}$/);
  assert.match(second, /^ok_[0-9a-f]{64}$/);
  assert.notEqual(first, second);
});

test("invalid API keys fail before startup with a generic error", () => {
  for (const apiKey of ["", " ", "secret token", "secret\tvalue", "secret\n", "secret\rvalue",
    "secret\0value", "secret\x7fvalue", "sëcret", null, 123]) {
    assert.throws(() => new OpenKindServer({ binary: "/nonexistent/openkindd", apiKey }), {
      name: "TypeError", message: "apiKey must be nonempty visible ASCII without whitespace",
    });
  }
  for (const apiKey of ["!", "~", "-secret", generateApiKey()]) {
    assert.doesNotThrow(() => new OpenKindServer({ apiKey }));
  }
});

test("optional keys reach the child only through its isolated environment", async () => {
  const directory = await mkdtemp(join(tmpdir(), "openkind-ts-key-"));
  const binary = join(directory, "inspect-daemon");
  const capture = join(directory, "child.json");
  const sources = ["OPENKIND_API_KEY", "OPENDECISION_API_KEY", "OPENPICK_API_KEY", "TYPESAFE_API_KEY"];
  const inherited = Object.fromEntries(sources.map((source) => [source, process.env[source]]));
  await writeFile(binary, `#!/usr/bin/env python3
import json, os, sys
with open(sys.argv[1], "w") as output:
    json.dump({"argv": sys.argv[2:], "credentials": {source: os.environ.get(source) for source in ${JSON.stringify(sources)}}}, output)
os.execv(sys.argv[2], sys.argv[2:])
`, { mode: 0o755 });
  try {
    for (const source of sources) process.env[source] = "inherited-secret";
    for (const apiKey of [undefined, generateApiKey()]) {
      const server = new OpenKindServer({ binary, extraArgs: [capture, fixture], apiKey,
        httpAddr: `127.0.0.1:${await freePort()}`, models: ["mock"] });
      await server.start();
      try {
        const child = JSON.parse(await readFile(capture, "utf8"));
        assert.deepEqual(child.credentials, { OPENKIND_API_KEY: apiKey ?? null,
          OPENDECISION_API_KEY: null, OPENPICK_API_KEY: null, TYPESAFE_API_KEY: null });
        assert.equal(child.argv.includes("--api-key"), false);
        if (apiKey !== undefined) assert.equal(child.argv.some((arg) => arg.includes(apiKey)), false);
        assert.equal(server.apiKey, apiKey);
        assert.equal((await server.client().listModels()).data.models[0].name, "mock");
        assert.equal((await fetch(`${server.baseUrl}/v1/models`)).status, apiKey === undefined ? 200 : 401);
        assert.equal((await fetch(`${server.baseUrl}/health`)).status, 200);
      } finally {
        await server.stop();
      }
    }
  } finally {
    for (const source of sources) {
      if (inherited[source] === undefined) delete process.env[source];
      else process.env[source] = inherited[source];
    }
    await rm(directory, { recursive: true, force: true });
  }
});

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

test("concurrent starts cannot replace the owned process", async () => {
  const server = new OpenKindServer({ binary: fixture, httpAddr: `127.0.0.1:${await freePort()}` });
  const first = server.start();
  await assert.rejects(server.start(), { name: "ServerError", message: "server has already been started" });
  try {
    await first;
    assert.equal(server.running, true);
    assert.equal((await server.client().health()).data.status, "ok");
  } finally {
    await server.stop();
  }
  assert.equal(server.running, false);
});

test("stop during the port probe cancels startup before spawning", async () => {
  const server = new OpenKindServer({ binary: fixture, httpAddr: `127.0.0.1:${await freePort()}` });
  const starting = server.start();
  const rejected = assert.rejects(starting, { name: "ServerError", message: "openkindd startup was cancelled" });
  await server.stop();
  await rejected;
  assert.equal(server.running, false);
  await server.start();
  await server.stop();
});

test("caller option mutations cannot change validated daemon arguments", async () => {
  const models = ["mock"];
  const extraArgs = [];
  const server = new OpenKindServer({ binary: fixture, httpAddr: `127.0.0.1:${await freePort()}`, models, extraArgs });
  models[0] = "changed";
  extraArgs.push("--help");
  await server.start();
  try {
    assert.equal((await server.client().listModels()).data.models[0].name, "mock");
    await Promise.all([server.stop(), server.stop()]);
    assert.equal(server.running, false);
  } finally {
    await server.stop();
  }
});

test("server wrapper starts a real openkindd binary and restarts on the same address", {
  skip: !process.env.OPENKIND_TEST_BINARY,
}, async () => {
  const server = new OpenKindServer({ binary: process.env.OPENKIND_TEST_BINARY,
    httpAddr: `127.0.0.1:${await freePort()}`, models: ["mock"], apiKey: generateApiKey() });
  for (let cycle = 0; cycle < 2; cycle++) {
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
  }
});
