# Jev wire and SDK compatibility

Checked 23 September 2026 against [TypeSafe's live OpenAPI](https://api.typesafe.ai/openapi.json), the [TypeSafe API guide](https://docs.typesafe.ai/api), the [TypeScript SDK](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/types.ts), the [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python/blob/0ffd094c72ed9445223060b24ffd7a56aa781fb4/src/typesafe_sdk/_core/endpoints.py), [OpenRouter's System One guide](https://openrouter.ai/docs/guides/community/typesafe-sdk), and Cloudflare's [Jev input](https://developers.cloudflare.com/ai/models/typesafe/jev/schema-input.json), [output](https://developers.cloudflare.com/ai/models/typesafe/jev/schema-output.json), and [REST example](https://developers.cloudflare.com/ai/models/typesafe/jev/). The schemas describe accepted shapes, not verified live behavior for every edge case. No authenticated provider calls were made.

## Transport and model selection

| Surface | Endpoint and request | Model handling | Response |
|---|---|---|---|
| OpenKind, TypeSafe | `POST /v1/systemone` with `state`, `model`, `questions` | `model` required on the wire | `model`, `answers`, `usage` |
| OpenRouter System One | `POST https://openrouter.ai/api/v1/systemone`, same required fields | Bare `jev-1.13` maps to `typesafe/jev-1.13`; `jev-latest` maps to the OpenRouter latest alias. A qualified ID is used as supplied. | Core response plus `id`, `provider`, `usage.cost` |
| OpenRouter Decisions | `POST https://openrouter.ai/api/alpha/decisions` | Qualified OpenRouter model ID | A separate API surface with routing and tracing fields. The Rust client does not target it. |
| Cloudflare Workers binding | `env.AI.run('typesafe/jev', { state, questions })` | Model is an argument to `run`, outside the input schema | Jev output object |
| Cloudflare REST | `POST /client/v4/accounts/{id}/ai/run` with `{ "model": "typesafe/jev", "input": { "state": ..., "questions": ... } }` | Model is in the outer REST envelope | Cloudflare REST `result` contains the Jev output object |

The official TypeScript `SystemOneRequest.model` is optional to callers, but its HTTP payload model is required. The client inserts `request.model ?? defaultModel`. The Python SDK likewise uses `default_model` when `model` is omitted. `openkind-client::Client::system_one` follows that pattern; `SystemRequest` and `Client::evaluate` require a model. Do not remove `model` from the OpenKind HTTP schema to imitate the SDK method signature.

The official SDK types do not perfectly mirror the live TypeSafe OpenAPI. The [TypeScript types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/types.ts#L682-L775) allow `null` for root state, while the [Python `JSONContent` type](https://github.com/typesafe-ai/typesafe-sdk-python/blob/0ffd094c72ed9445223060b24ffd7a56aa781fb4/src/typesafe_sdk/_core/json_types.py#L247-L263) and live OpenAPI do not. TypeScript's Score tuple requires at least two levels and allows `null` entries. Python's Score type is a sequence of non-null string/object/array values, while the live OpenAPI has a minimum of one. Both SDKs permit omitted instructions, matching the TypeSafe OpenAPI; Cloudflare's model schema requires the key. These are declaration differences, not proof that the hosted service accepts every value the SDK can represent.

## Field comparison

The OpenKind column describes the current Rust types and request/response validation. The TypeSafe column uses the live OpenAPI as machine-readable authority; its prose sometimes states tighter guidance. OpenRouter System One documents the TypeSafe shape. Cloudflare's columns describe the model input/output schemas, inside its transport envelope.

| Field | TypeSafe / OpenRouter System One | Cloudflare Jev | OpenKind server and Rust SDK |
|---|---|---|---|
| `state` | Required string, object, or array | Required string, object, array, or `null` | Required string, object, or array |
| `model` | Required string in HTTP body | Absent from model input; set in binding call or REST envelope | Required string in HTTP body; client default `jev-latest` |
| `questions` | Required map, at least one entry | Required map; schema does not set a minimum | Required map; runtime requires 1 to 10000 entries |
| Question IDs | String keys | Nonempty string keys | String keys; IDs are preserved in `answers` |
| Question `type` | `noul`, `choice`, `score` | Same | Same; unknown types fail decoding |
| `instructions` | Optional; string, object, array, or `null` | Required; string, object, array, or `null` | Required; runtime rejects null or empty string/object/array. Parser currently accepts number and boolean, which are outside the provider schemas. |
| Noul `criteria` | Optional or `null`; `true`/`false` each optional, with string/object/array/null values | Same, with a closed criteria object | Optional or `null`; if present, both `true` and `false` must be nonempty strings |
| Choice `criteria` | Required map; values string/object/array/null | Same | Required nonempty map; values string/null; runtime maximum 10000 options |
| Score `criteria` | Required array, at least 1; items string/object/array | Required array, at least 2; items string/object/array/null | Required array of nonempty strings, 2 to 10000 levels |
| Noul answer | Required `type`, `noul`; no confidence | Same, with probability bound 0 to 1 | Same; runtime bounds probability to 0 to 1 |
| Choice answer | Required `type`, `choice`, `confidence`, `probabilities` | Same, with numeric bounds | Same; runtime checks range, sum, selected key, and probability keys against the request |
| Score answer | Required `type`, `score`, `legend`, `confidence`, `probabilities`; legend values may be string/object/array | Same fields, but legend values are strings | Same fields; legend values are strings; runtime checks range, sum, and legend/probability keys |
| `usage` | Required integer `input_tokens`, `output_tokens`; no upper bound in live OpenAPI | Required integers bounded to 2^53-1 | Required `u32` integers, up to 2^32-1 |
| Extra fields | Live TypeSafe OpenAPI leaves objects open | Cloudflare model schemas close top-level and question/answer objects | OpenKind ignores extra JSON fields on decode. OpenRouter's `id`, `provider`, and `usage.cost` are accepted but discarded by the Rust response type. |

TypeSafe's [prose guide](https://docs.typesafe.ai/api) gives a Score maximum of 10 levels and Choice maximum of 255 options. Its live OpenAPI does not encode those maxima and allows one Score level; Cloudflare requires two. The OpenKind OpenAPI describes its implemented 10000-option admission ceiling, not the upstream prose limit. Portable Score calls should use 2 to 10 nonempty string levels, and portable Choice calls should stay within 255 options.

The native Qwen profile requires an explicit `__none__` Choice option. That is a backend requirement, not a global Jev schema rule. See the [runtime architecture](ARCHITECTURE.md) and [server API](../crates/openkind-api/openapi.yaml) for local behavior.

## Rust client support

`openkind-client` uses the common string-criteria subset for all three question types. Set the endpoint, token, and default model for each provider:

| Provider | Builder settings | Supported evaluation path | Other methods |
|---|---|---|---|
| OpenKind | `.base_url("http://127.0.0.1:18080")`, local token, registered model | `/v1/systemone` | `list_models`, `health` |
| TypeSafe | Default base URL, TypeSafe key, default `jev-latest` | `/v1/systemone` | `list_models`; `health` is OpenKind-specific |
| OpenRouter | `.base_url("https://openrouter.ai/api")`, OpenRouter key, `.default_model("jev-1.13")` | `/v1/systemone` | `list_models` is not portable: OpenRouter's `/api/v1/models` returns its own shape. `health` is unavailable. |
| Cloudflare | `.cloudflare_account(account_id)`, Cloudflare API token; default model `typesafe/jev` | `/ai/run`, with `input` wrapping and `result` unwrapping | `list_models` and `health` return configuration errors. |

The client validates returned answers against the original questions. Its current strict response types cannot decode TypeSafe's structured Score legend or token counts above `u32::MAX`. Cloudflare's raw model schema may accept inputs that the Rust request types cannot represent. These are genuine portability gaps, not evidence that providers reject the common subset. OpenRouter's extra response metadata is not exposed by `SystemResponse`.

Bearer tokens are required for each hosted provider, but error envelopes and retry headers are provider-specific. The Rust client's status-based retry rules still apply; its parsed error code, message, and request ID are only guaranteed for OpenKind's documented error shape. Provider response examples and local stub tests establish request routing and decoding, not authenticated live interoperability.

The OpenKind [OpenAPI document](../crates/openkind-api/openapi.yaml) specifies the local server, and the generated [request](../crates/openkind-core/schemas/jev-v1-request.json) and [response](../crates/openkind-core/schemas/jev-v1-response.json) schemas reflect Rust wire types. They are not a union of every provider's accepted shapes. In particular, Cloudflare's model input schema must not be substituted for `/v1/systemone` because it has no body `model` field.

The generated JSON Schemas currently describe Serde decoding, so they do not encode every runtime admission rule, such as nonempty instructions, Score cardinality, or probability normalization. The OpenAPI document states the local request limits that can be expressed as schema constraints. Runtime validators remain authoritative for cross-field checks.
