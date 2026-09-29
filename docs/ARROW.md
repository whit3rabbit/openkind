# Arrow bulk endpoint (unofficial)

OpenKind exposes an opt-in bulk evaluation endpoint, `POST /v1/arrow`, that
answers many independent states against one shared question map and returns
the decisions as an [Apache Arrow](https://arrow.apache.org/) IPC stream:
one row per state, one column per question. The design follows the
proposal in [“What if Jev spoke Arrow?”](https://columnar.tech/blog/what-if-jev-spoke-arrow/).

**This endpoint is unofficial.** It is not part of the TypeSafe Jev wire
contract, it is intentionally absent from
[`openapi.yaml`](../crates/openkind-api/openapi.yaml), and it is disabled by
default. The TypeSafe-compatible surface (`POST /v1/systemone`,
`POST /v1/system_one`, `GET /v1/models`, and the gRPC `SystemOne` service)
is unchanged; see [`JEV_COMPATIBILITY.md`](JEV_COMPATIBILITY.md) for the
compatibility matrix.

## Enabling

The endpoint is served only when the daemon opts in:

```bash
openkindd --arrow on          # or: OPENKIND_ARROW=on
```

Without the flag the route is absent (404). With it, the endpoint joins the
standard `/v1/*` middleware stack: bearer authentication when an API key is
configured, per-IP rate limiting, and `x-typesafe-request-id` response
stamping all apply.

## Request

`Content-Type: application/json`. The body mirrors the
[`SystemRequest`](../crates/openkind-api/openapi.yaml) shape with `state`
generalized to a `states` array:

```json
{
  "model": "jev-latest",
  "states": ["Please refund the shoes.", {"cart": ["shoes", "socks"]}],
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund being requested?"},
    "department": {
      "type": "choice",
      "instructions": "Which department does this belong to?",
      "criteria": {"returns": "Returns and exchanges", "shipping": "Shipping enquiries", "__none__": "Neither department applies"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Can wait", "Within a few days", "Today"]
    }
  }
}
```

- `model`, `states`, and `questions` are required. An empty `questions` map
  is rejected (422), matching single-state validation; an empty `states`
  array is accepted and yields a zero-row batch. Questions and the model
  are still validated; an unknown model returns 404.
- Unknown top-level fields are ignored, exactly as on `/v1/systemone`.
- States are evaluated sequentially against one pinned engine handle. Results
  keep input order, and the first failure stops evaluation of later states.
- Native Choice models require a non-empty `__none__` option description;
  include it in the shared criteria map. It counts toward the 256-option cap.

## Response

`Content-Type: application/vnd.apache.arrow.stream`. The body is a
schema-first [Arrow IPC stream](https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc):
a schema, one record batch, and an end-of-stream marker. It is readable by PyArrow's
`ipc.open_stream`, and consumable directly by Polars, DuckDB, DataFusion, and
any other Arrow engine.

### Columns

Each question id becomes a column; columns are ordered by question id
(lexicographic). Row `i` holds the answer for `states[i]`. Every field is
non-nullable: the endpoint evaluates every state or fails the whole request
before any Arrow bytes are written.

| Jev type | Arrow type | Field metadata |
|---|---|---|
| Noul | `float64` | `jev.type = "noul"` |
| Choice | `struct<choice: uint8, confidence: float64, probabilities: fixed_size_list<float64>[N]>` | `jev.type = "choice"`, `labels` |
| Score | `struct<score: float64, confidence: float64, probabilities: fixed_size_list<float64>[N]>` | `jev.type = "score"`, `legend` |

Field metadata values are strings; `labels` and `legend` carry a JSON-encoded
array of strings.

- `labels` lists the Choice option keys in sorted order (the wire criteria
  map is unordered, so sorted order is the deterministic projection). The
  `choice` child is the index into that array. `probabilities` entries align
  positionally with `labels`.
- `legend` lists the Score level descriptions in rubric order (the order of
  the question's `criteria` array). `probabilities` entries and the JSON
  answer's `"0"`, `"1"`, … keys align with those positions; `score` is the
  expected value unchanged.
- `confidence` is the answer's confidence unchanged.

The values plus metadata are sufficient to reconstruct the original answer
objects. `openkind_api::arrow::answers_from_batch` validates version 1,
non-null fields and values, metadata, and Jev numeric rules before returning
answers. Round-trip and malformed-batch tests pin this behavior.

### Schema metadata

| Key | Meaning |
|---|---|
| `openkind.arrow.version` | Format version of this mapping (`1`). |
| `openkind.model` | Model that performed the evaluation; the requested alias when `states` is empty. |
| `openkind.usage.input_tokens` | Aggregate input tokens across all state evaluations. |
| `openkind.usage.output_tokens` | Aggregate output tokens across all state evaluations. |

## Errors

Failures are all-or-nothing and use the standard JSON error envelope and
status taxonomy (`invalid_body` 422, `unknown_model` 404, `rate_limited`
429, `overloaded` 529, `deadline_exceeded` 504). The schema is emitted
only after every state has been evaluated, so a reader either gets a
complete stream or a JSON error body.

## Limits

- At most 10,000 states per request; larger batches must be chunked.
- Choice questions support at most 256 options: the `choice` child is a
  `uint8` index. `/v1/systemone` allows more options; such questions are
  rejected here with 422.
- The standard 16 MiB request payload limit applies.
- Projected numeric column buffers and the complete encoded IPC response
  must each fit within 64 MiB. Oversized projections are rejected before
  evaluation with 422; the encoder also enforces the complete response cap.
- The complete batch evaluation and encoding has a 600-second deadline. A
  deadline failure returns 504. Each backend may enforce a shorter deadline.
  Chunk large workloads to stay within the byte and time limits.
- Rate limiting counts one HTTP request per call, so a bulk request moves
  many state evaluations behind a single rate-limit unit.

## Consuming from Python

```python
import json
import httpx
import pyarrow.ipc as ipc

resp = httpx.post(
    "http://127.0.0.1:8080/v1/arrow",
    headers={"Authorization": "Bearer <key>"},
    timeout=660.0,  # Allow the batch deadline plus transport overhead.
    json={
        "model": "jev-latest",
        "states": ["Please refund the shoes.", "Where is my parcel?"],
        "questions": {
            "refund": {"type": "noul", "instructions": "Is a refund being requested?"},
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"returns": "Returns and exchanges", "shipping": "Shipping enquiries", "__none__": "Neither department applies"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Can wait", "Today"],
            },
        },
    },
)
resp.raise_for_status()

with ipc.open_stream(resp.content) as reader:
    print(reader.schema.metadata)  # b"openkind.model", usage keys, ...
    table = reader.read_all()

for row in table.to_pylist():
    for name, field in zip(table.column_names, table.schema):
        jev_type = field.metadata[b"jev.type"].decode()
        if jev_type == "noul":
            row[name] = {"noul": row[name]}
        elif jev_type == "choice":
            labels = json.loads(field.metadata[b"labels"])
            row[name]["choice"] = labels[row[name]["choice"]]
            row[name]["probabilities"] = dict(zip(labels, row[name]["probabilities"]))
        elif jev_type == "score":
            legend = json.loads(field.metadata[b"legend"])
            keys = [str(i) for i in range(len(legend))]
            row[name]["legend"] = dict(zip(keys, legend))
            row[name]["probabilities"] = dict(zip(keys, row[name]["probabilities"]))
    print(row)
```

Score keys identify positions, so repeated descriptions remain separate
levels in both the legend and probabilities.

## Implementation

The mapping lives in [`crates/openkind-api/src/arrow.rs`](../crates/openkind-api/src/arrow.rs)
with unit tests in `arrow_tests.rs`, HTTP integration tests in `http_tests.rs`,
and the daemon flag in [`crates/openkind-server/src/args.rs`](../crates/openkind-server/src/args.rs).
Numeric column builders retain values between evaluations and discard each
complete engine response. IPC encoding runs on a blocking worker. The HTTP
body is returned only after encoding finishes, so a reader cannot observe a
partially evaluated batch. The Rust router API preserves `router_daemon` with
Arrow disabled and adds `router_daemon_with_arrow` for explicit opt-in.
