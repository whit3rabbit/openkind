# TypeScript client

Node 18+ or another runtime with `fetch`. Build from this directory with `npm install && npm run build`; run `npm test` for local contract tests. The package is source only in this repository and has not been published to npm.

```ts
import { OpenKindClient } from "@openkind/client";

const client = new OpenKindClient({ baseUrl: "http://127.0.0.1:18080" });
const result = await client.systemOne("I was charged twice.", {
  billing: { type: "noul", instructions: "Is this a billing issue?" },
}, "mock");
console.log(result.data.answers.billing, result.requestId);
```

`evaluate(request)` accepts an explicit `SystemRequest`, `listModels()` lists registered aliases, and `health()` probes the daemon without auth. `ApiError` carries the HTTP status, error code, and request ID. `InvalidResponseError` marks malformed or request-inconsistent success responses. See the [shared scope](../README.md).

Use the key configured on the daemon as `apiKey`. Omit it when that daemon runs without authentication. A generated key takes effect only when the daemon receives the same value.

## Start a local server from Node

Build `openkindd` first, then use the Node-only `@openkind/client/server` entrypoint:

```ts
import { generateApiKey, OpenKindServer } from "@openkind/client/server";

const apiKey = generateApiKey();
const server = new OpenKindServer({
  binary: "/path/to/openkindd",
  httpAddr: "127.0.0.1:18080",
  models: ["mock"],
  apiKey,
});
await server.start();
try {
  console.log((await server.client().listModels()).data.models);
} finally {
  await server.stop();
}
```

`generateApiKey()` uses Node's operating-system randomness to return `ok_` followed by 64 lowercase hexadecimal characters (256 bits). It runs without a daemon subprocess and does not save the key. Keep the same value if you want to reuse it after restarting.

Authentication is optional. Omit `ServerOptions.apiKey` for an unauthenticated local benchmark or test server; inherited credential variables are cleared. Supplied keys must be nonempty visible ASCII without whitespace. `server.client()` uses the configured key, which reaches the daemon through its environment rather than its arguments.

The main client entrypoint works with any runtime that provides `fetch`. The server entrypoint uses Node process APIs and crypto. It manages only the child it launches, with HTTP on loopback and gRPC disabled.

Set `OPENKIND_TEST_URL` and, if needed, `OPENKIND_TEST_API_KEY` to include the live daemon test. Set `OPENKIND_TEST_BINARY` to an absolute `openkindd` path to test local server startup and restart with a generated key.
