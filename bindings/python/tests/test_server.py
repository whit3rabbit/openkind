import os
import socket
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from openkind_client import Server, ServerError


FIXTURE = Path(__file__).parents[2] / "test" / "fake_openkindd.py"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerTests(unittest.TestCase):
    def test_process_lifecycle_and_client(self):
        server = Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}",
                        models=("mock",), api_key="secret")
        with server:
            self.assertTrue(server.running)
            self.assertEqual(server.client.health().data["status"], "ok")
            self.assertEqual(server.client.list_models().data["models"][0]["name"], "mock")
            result = server.client.system_one("billing ticket", {
                "billing": {"type": "noul", "instructions": "Is this billing?"},
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": None, "sales": None}},
                "severity": {"type": "score", "instructions": "Rate severity",
                             "criteria": ["low", "high"]},
            }, model="mock")
            self.assertEqual(result.data["answers"]["billing"]["type"], "noul")
            self.assertEqual(set(result.data["answers"]), {"billing", "team", "severity"})
            with self.assertRaises(HTTPError) as unauthorized:
                urlopen(server.base_url + "/v1/models")
            self.assertEqual(unauthorized.exception.code, 401)
            unauthorized.exception.close()
        self.assertFalse(server.running)

    def test_refuses_occupied_address(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            address = f"127.0.0.1:{occupied.getsockname()[1]}"
            server = Server(binary=str(FIXTURE), http_addr=address)
            with self.assertRaises(ServerError):
                server.start()
            self.assertFalse(server.running)

    def test_reports_missing_executable(self):
        server = Server(binary="/nonexistent/openkindd", http_addr=f"127.0.0.1:{free_port()}")
        with self.assertRaises(ServerError):
            server.start()
        self.assertFalse(server.running)

    @unittest.skipUnless(os.getenv("OPENKIND_TEST_BINARY"), "set OPENKIND_TEST_BINARY for real openkindd")
    def test_real_binary(self):
        server = Server(binary=os.environ["OPENKIND_TEST_BINARY"],
                        http_addr=f"127.0.0.1:{free_port()}", models=("mock",), api_key="dev-key")
        with server:
            self.assertEqual(server.client.health().data["status"], "ok")
            result = server.client.system_one("billing ticket", {
                "billing": {"type": "noul", "instructions": "Is this billing?"},
            }, model="mock")
            self.assertEqual(result.data["answers"]["billing"]["type"], "noul")
        self.assertFalse(server.running)


if __name__ == "__main__":
    unittest.main()
