import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openkind_client import ApiError, Client, InvalidResponseError


QUESTION = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "sales": "Sales"}}


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, _format, *_args):
        pass

    def respond(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("x-typesafe-request-id", "request-123")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, self.headers.get("Authorization"), body))
        if self.path != "/v1/systemone":
            self.respond(404, {"error": {"code": "not_found", "message": "missing"}})
        elif self.headers.get("Authorization") != "Bearer test-key":
            self.respond(401, {"error": {"code": "unauthorized", "message": "no token"}})
        elif body["state"] == "rate":
            self.respond(429, {"error": {"code": "rate_limited", "message": "slow down"}})
        else:
            self.respond(200, {
                "model": "mock",
                "answers": {"team": {
                    "type": "choice", "choice": "outside" if body["state"] == "bad" else "billing",
                    "probabilities": {"billing": 0.75, "sales": 0.25}, "confidence": 0.75,
                }},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            })

    def do_GET(self):
        self.calls.append((self.path, self.headers.get("Authorization"), None))
        if self.path == "/health":
            self.respond(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.respond(200, {"models": [{
                "name": "mock", "description": "Mock engine", "release_date": "2026-01-01",
            }]})
        else:
            self.respond(404, {"error": {"code": "not_found", "message": "missing"}})


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.calls.clear()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}", api_key="test-key")

    def test_evaluate_and_request_body(self):
        result = self.client.system_one("ticket", {"team": QUESTION}, model="mock")
        self.assertEqual(result.data["answers"]["team"]["choice"], "billing")
        self.assertEqual(result.request_id, "request-123")
        self.assertEqual(Handler.calls[0], (
            "/v1/systemone", "Bearer test-key",
            {"state": "ticket", "model": "mock", "questions": {"team": QUESTION}},
        ))

    def test_rejects_out_of_set_choice(self):
        with self.assertRaises(InvalidResponseError):
            self.client.system_one("bad", {"team": QUESTION}, model="mock")

    def test_error_envelope(self):
        with self.assertRaises(ApiError) as caught:
            self.client.system_one("rate", {"team": QUESTION}, model="mock")
        self.assertEqual((caught.exception.status, caught.exception.code, caught.exception.request_id),
                         (429, "rate_limited", "request-123"))

    def test_models_and_health(self):
        self.assertEqual(self.client.list_models().data["models"][0]["name"], "mock")
        self.assertEqual(self.client.health().data["status"], "ok")
        self.assertEqual(Handler.calls, [
            ("/v1/models", "Bearer test-key", None),
            ("/health", None, None),
        ])

    @unittest.skipUnless(os.getenv("OPENKIND_TEST_URL"), "set OPENKIND_TEST_URL for a live daemon")
    def test_live_daemon(self):
        client = Client(base_url=os.environ["OPENKIND_TEST_URL"],
                        api_key=os.getenv("OPENKIND_TEST_API_KEY", ""))
        result = client.system_one("A customer was charged twice.", {
            "billing": {"type": "noul", "instructions": "Is this a billing issue?"},
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": None, "sales": None}},
            "severity": {"type": "score", "instructions": "Rate severity",
                         "criteria": ["low", "high"]},
        }, model="mock")
        self.assertEqual(result.data["model"], "mock")
        self.assertIsNotNone(result.request_id)
        self.assertEqual(set(result.data["answers"]), {"billing", "team", "severity"})
        self.assertIn("mock", [model["name"] for model in client.list_models().data["models"]])
        self.assertEqual(client.health().data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
