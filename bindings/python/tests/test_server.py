import os
import socket
import subprocess
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest import mock

from openkind_client import Server, ServerError


FIXTURE = Path(__file__).parents[2] / "test" / "fake_openkindd.py"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerTests(unittest.TestCase):
    def test_does_not_inherit_api_keys(self):
        aliases = ("OPENKIND_API_KEY", "OPENDECISION_API_KEY", "OPENPICK_API_KEY", "TYPESAFE_API_KEY")
        with mock.patch.dict(os.environ, {name: "inherited-secret" for name in aliases}):
            with mock.patch("openkind_client.server.subprocess.Popen", wraps=subprocess.Popen) as spawn:
                with Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}", models=("mock",)) as server:
                    self.assertEqual(server.client.list_models().data["models"][0]["name"], "mock")
                self.assertTrue(all(name not in spawn.call_args.kwargs["env"] for name in aliases))

    def test_start_interruption_stops_owned_child(self):
        server = Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}")
        processes = []
        spawn = subprocess.Popen

        def capture_process(*args, **kwargs):
            process = spawn(*args, **kwargs)
            processes.append(process)
            return process

        with mock.patch("openkind_client.server.subprocess.Popen", side_effect=capture_process):
            with mock.patch("openkind_client.server.urlopen", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    server.start()
        self.assertFalse(server.running)
        self.assertIsNone(server._process)
        self.assertIsNotNone(processes[0].poll())

    def test_stop_reaps_child_that_exits_during_termination(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.terminate.side_effect = ProcessLookupError
        server = Server(http_addr=f"127.0.0.1:{free_port()}")
        server._process = process
        server.stop()
        process.wait.assert_called_once_with(timeout=server.shutdown_timeout)
        self.assertIsNone(server._process)

    def test_stop_retains_child_after_failed_termination(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.terminate.side_effect = PermissionError
        server = Server(http_addr=f"127.0.0.1:{free_port()}")
        server._process = process
        with self.assertRaises(PermissionError):
            server.stop()
        self.assertIs(server._process, process)

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
