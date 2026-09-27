# TypeScript client

Node 18+ or another runtime with `fetch`. Build from this directory with `npm install && npm run build`; run `npm test` for local contract tests. The package is source only in this repository and has not been published to npm.

```ts
import { OpenKindClient } from "@openkind/client";

const client = new OpenKindClient({ baseUrl: "http://127.0.0.1:18080", apiKey: "dev-key" });
const result = await client.systemOne("I was charged twice.", {
  billing: { type: "noul", instructions: "Is this a billing issue?" },
}, "mock");
console.log(result.data.answers.billing, result.requestId);
```

`evaluate(request)` accepts an explicit `SystemRequest`, `listModels()` lists registered aliases, and `health()` probes the daemon without auth. `ApiError` carries the HTTP status, error code, and request ID. `InvalidResponseError` marks malformed or request-inconsistent success responses. See the [shared scope](../README.md).

## Start a local server from Node

Build `openkindd` first, then use the Node-only `@openkind/client/server` entrypoint:

```ts
import { OpenKindServer } from "@openkind/client/server";

const server = new OpenKindServer({
  binary: "/path/to/openkindd",
  httpAddr: "127.0.0.1:18080",
  models: ["mock"],
  apiKey: "dev-key",
});
await server.start();
try {
  console.log((await server.client().listModels()).data.models);
} finally {
  await server.stop();
}
```

The main client entrypoint works with any runtime that provides `fetch`. The server entrypoint uses Node process APIs. It manages only the child it launches, with HTTP on loopback and gRPC disabled.
Set `OPENKIND_TEST_URL` and, if needed, `OPENKIND_TEST_API_KEY` to include the live daemon test.
