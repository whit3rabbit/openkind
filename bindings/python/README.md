# Python client

Requires Python 3.11+. Install locally with `python3 -m pip install ./bindings/python` from the repository root, or set `PYTHONPATH=bindings/python`. Run `python3 -m unittest discover -s tests` from this directory for local contract tests. This package has not been published to PyPI.

```python
from openkind_client import Client

client = Client(base_url="http://127.0.0.1:18080", api_key="dev-key")
result = client.system_one("I was charged twice.", {
    "billing": {"type": "noul", "instructions": "Is this a billing issue?"},
}, model="mock")
print(result.data["answers"]["billing"], result.request_id)
```

`evaluate(request)` accepts an explicit `SystemRequest`, `list_models()` lists registered aliases, and `health()` probes the daemon without auth. `ApiError` carries the HTTP status, error code, and request ID. `InvalidResponseError` marks malformed or request-inconsistent success responses. When arguments are omitted, the client uses `OPENKIND_BASE_URL`, `OPENKIND_API_KEY`, and `OPENKIND_DEFAULT_MODEL`, falling back to their `TYPESAFE_*` counterparts. See the [shared scope](../README.md).

## Start a local server

Build `openkindd` first. The `Server` context manager stops its child on exit:

```python
from openkind_client import Server

with Server(binary="/path/to/openkindd", http_addr="127.0.0.1:18080",
            models=("mock",), api_key="dev-key") as server:
    print(server.client.list_models().data["models"])
```

`Server` binds HTTP to loopback, disables gRPC, and does not control an existing daemon.
Set `OPENKIND_TEST_URL` and, if needed, `OPENKIND_TEST_API_KEY` to include the live daemon test.
