#!/usr/bin/env python3
"""Small process fixture for language binding lifecycle tests."""

import argparse
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


parser = argparse.ArgumentParser()
parser.add_argument("--http-addr", required=True)
parser.add_argument("--grpc-addr", required=True)
parser.add_argument("--models", required=True)
args, _ = parser.parse_known_args()
if args.grpc_addr != "0":
    parser.error("tests require gRPC disabled")
host, port = args.http_addr.rsplit(":", 1)
models = args.models.split(",")
api_key = os.getenv("OPENKIND_API_KEY")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-typesafe-request-id", "fixture-request")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
            self.reply(401, {"error": {"code": "unauthorized", "message": "invalid key"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok"})
        elif self.path == "/v1/models":
            if not self.authorized():
                return
            self.reply(200, {"models": [
                {"name": name, "description": "Fixture model", "release_date": "2026-01-01"}
                for name in models
            ]})
        else:
            self.reply(404, {"error": {"code": "not_found", "message": "missing"}})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.reply(404, {"error": {"code": "not_found", "message": "missing"}})
            return
        if not self.authorized():
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answers = {}
        for name, question in request["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.75}
            elif question["type"] == "choice":
                options = list(question["criteria"])
                answers[name] = {
                    "type": "choice", "choice": options[0], "confidence": 1.0,
                    "probabilities": {key: float(key == options[0]) for key in options},
                }
            else:
                levels = question["criteria"]
                answers[name] = {
                    "type": "score", "score": 0.0, "confidence": 1.0,
                    "legend": {str(index): label for index, label in enumerate(levels)},
                    "probabilities": {str(index): float(index == 0) for index in range(len(levels))},
                }
        self.reply(200, {"model": request["model"], "answers": answers,
                         "usage": {"input_tokens": 1, "output_tokens": 1}})


server = ThreadingHTTPServer((host, int(port)), Handler)
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
server.serve_forever(poll_interval=0.05)
server.server_close()
