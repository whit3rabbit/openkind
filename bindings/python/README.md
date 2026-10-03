# Python client

Requires Python 3.11+. Install locally with `python3 -m pip install ./bindings/python` from the repository root, or set `PYTHONPATH=bindings/python`. Run `python3 -m unittest discover -s tests` from this directory for local contract tests. This package has not been published to PyPI.

```python
from openkind_client import Client

client = Client(base_url="http://127.0.0.1:18080")
result = client.system_one("I was charged twice.", {
    "billing": {"type": "noul", "instructions": "Is this a billing issue?"},
}, model="mock")
print(result.data["answers"]["billing"], result.request_id)
```

`evaluate(request)` accepts an explicit `SystemRequest`, `list_models()` lists registered aliases, and `health()` probes the daemon without auth. `ApiError` carries the HTTP status, error code, and request ID. `InvalidResponseError` marks malformed or request-inconsistent success responses. When arguments are omitted, the client uses `OPENKIND_BASE_URL`, `OPENKIND_API_KEY`, and `OPENKIND_DEFAULT_MODEL`, falling back to their `TYPESAFE_*` counterparts. See the [shared scope](../README.md).

Authentication is optional. For an existing local or remote daemon, define its
key with `--api-key` or `OPENKIND_API_KEY` and give the client the same value:

```python
client = Client(base_url="https://your-server.example", api_key="configured-server-key")
```

## Start a local server

Build `openkindd` first. The `Server` context manager stops its child on exit.
Omitting `api_key` leaves authentication off, including when the parent process
has API key environment variables. To require a key, generate one explicitly:

```python
from openkind_client import Server, generate_api_key

api_key = generate_api_key()
with Server(binary="/path/to/openkindd", http_addr="127.0.0.1:18080",
            models=("mock",), api_key=api_key) as server:
    print(server.client.list_models().data["models"])
```

`generate_api_key()` uses 32 random bytes and returns `ok_` followed by 64
lowercase hexadecimal characters. Generation does not enable authentication,
and the package does not save keys. Retain the returned value yourself to reuse it.

`Server` accepts nonempty visible ASCII keys without whitespace. It passes the
key through the child environment and provides a client with the matching key.
The wrapper binds HTTP to loopback, disables gRPC, and does not control an
existing daemon.

Set `OPENKIND_TEST_URL` and, if needed, `OPENKIND_TEST_API_KEY` to include the
live daemon test. Set `OPENKIND_TEST_BINARY` to an `openkindd` executable to
include local wrapper tests with generated keys and authentication off.
