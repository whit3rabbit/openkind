import io
import os
import socket
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest import mock

from openkind_client import Server, ServerError, generate_api_key


FIXTURE = Path(__file__).parents[2] / "test" / "fake_openkindd.py"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerTests(unittest.TestCase):
    def test_invalid_api_keys_fail_before_launch_without_leaking_values(self):
        for key in ("", "sensitive key", "sensitive\tkey", "sensitive\nkey",
                    "sensitive\x00key", "sensitive\x7fkey", "sensitive-kéy"):
            with self.subTest(key=key):
                with mock.patch("openkind_client.server.subprocess.Popen") as spawn:
                    with self.assertRaises(ValueError) as error:
                        Server(api_key=key)
                    self.assertNotIn("sensitive", str(error.exception))
                    spawn.assert_not_called()

    def test_mutated_invalid_api_key_is_checked_before_launch(self):
        server = Server()
        server.api_key = "sensitive invalid key"
        with mock.patch("openkind_client.server.subprocess.Popen") as spawn:
            with self.assertRaises(ValueError) as error:
                server.start()
            self.assertNotIn("sensitive", str(error.exception))
            spawn.assert_not_called()
        self.assertFalse(server.running)

    def test_generated_key_is_forwarded_only_in_child_environment(self):
        aliases = ("OPENKIND_API_KEY", "OPENDECISION_API_KEY", "OPENPICK_API_KEY", "TYPESAFE_API_KEY")
        key = generate_api_key()
        with mock.patch.dict(os.environ, {name: "inherited-secret" for name in aliases}):
            with mock.patch("openkind_client.server.subprocess.Popen", wraps=subprocess.Popen) as spawn:
                with Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}",
                            models=("mock",), api_key=key) as server:
                    self.assertEqual(server.client.api_key, key)
                    self.assertEqual(server.client.list_models().data["models"][0]["name"], "mock")
                child_args = spawn.call_args.args[0]
                child_env = spawn.call_args.kwargs["env"]
                self.assertNotIn(key, child_args)
                self.assertNotIn("--api-key", child_args)
                self.assertEqual(child_env["OPENKIND_API_KEY"], key)
                self.assertTrue(all(name not in child_env for name in aliases[1:]))
                self.assertTrue(all(os.environ[name] == "inherited-secret" for name in aliases))

    def test_readiness_does_not_follow_redirects(self):
        calls = []

        class RedirectHealth(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                pass

            def do_GET(self):
                calls.append(self.path)
                if self.path == "/health":
                    self.send_response(302)
                    self.send_header("Location", "/elsewhere")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        listener = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHealth)
        worker = threading.Thread(target=listener.serve_forever, daemon=True)
        worker.start()
        server = Server(http_addr=f"127.0.0.1:{listener.server_port}", startup_timeout=0.05)
        process = mock.Mock()
        process.poll.return_value = None
        real_socket = socket.socket
        probe = mock.MagicMock()

        def socket_factory(*args, **kwargs):
            # Only the launch probe is replaced; the health request uses a real socket.
            return real_socket(*args, **kwargs) if args or kwargs else probe

        try:
            with mock.patch("openkind_client.server.socket.socket", side_effect=socket_factory):
                with mock.patch("openkind_client.server.subprocess.Popen", return_value=process):
                    with self.assertRaises(ServerError):
                        server.start()
            self.assertTrue(calls)
            self.assertTrue(all(path == "/health" for path in calls))
            process.terminate.assert_called_once()
            self.assertFalse(server.running)
        finally:
            listener.shutdown()
            listener.server_close()
            worker.join()

    def test_concurrent_start_launches_only_one_process(self):
        server = Server(http_addr=f"127.0.0.1:{free_port()}")
        process = mock.Mock()
        process.poll.return_value = None
        launching = threading.Event()
        release_launch = threading.Event()
        second_lock_attempt = threading.Event()
        errors = []

        class ObservedLock:
            def __init__(self):
                self.lock = threading.RLock()

            def __enter__(self):
                if threading.current_thread().name == "second-start":
                    second_lock_attempt.set()
                self.lock.acquire()

            def __exit__(self, *_exc):
                self.lock.release()

        server._lock = ObservedLock()

        def launch(*_args, **_kwargs):
            launching.set()
            if not release_launch.wait(2):
                raise RuntimeError("test did not release launch")
            return process

        def start():
            try:
                server.start()
            except BaseException as error:
                errors.append(error)

        def healthy(*_args, **_kwargs):
            response = io.BytesIO(b'{"status":"ok"}')
            response.status = 200
            return response

        first = threading.Thread(target=start, name="first-start")
        second = threading.Thread(target=start, name="second-start")
        with mock.patch("openkind_client.server.subprocess.Popen", side_effect=launch) as spawn:
            with mock.patch("openkind_client.server._OPENER.open", side_effect=healthy):
                first.start()
                try:
                    self.assertTrue(launching.wait(2))
                    second.start()
                    self.assertTrue(second_lock_attempt.wait(2))
                    self.assertEqual(spawn.call_count, 1)
                finally:
                    release_launch.set()
                    first.join(2)
                    if second.ident is not None:
                        second.join(2)
                    server.stop()
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ServerError)
        self.assertEqual(spawn.call_count, 1)

    def test_cancelled_start_does_not_return_or_stop_replacement(self):
        server = Server(http_addr=f"127.0.0.1:{free_port()}")
        original = mock.Mock()
        replacement = mock.Mock()
        original.poll.return_value = replacement.poll.return_value = None
        probing = threading.Event()
        release_probe = threading.Event()
        errors = []

        def health(*_args, **_kwargs):
            if threading.current_thread().name == "original-start":
                probing.set()
                if not release_probe.wait(2):
                    raise RuntimeError("test did not release readiness")
            response = io.BytesIO(b'{"status":"ok"}')
            response.status = 200
            return response

        def start():
            try:
                server.start()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=start, name="original-start")
        with mock.patch("openkind_client.server.subprocess.Popen", side_effect=(original, replacement)):
            with mock.patch("openkind_client.server._OPENER.open", side_effect=health):
                worker.start()
                try:
                    self.assertTrue(probing.wait(2))
                    server.stop()
                    server.start()
                    release_probe.set()
                    worker.join(2)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], ServerError)
                    self.assertIs(server._process, replacement)
                    replacement.terminate.assert_not_called()
                finally:
                    release_probe.set()
                    worker.join(2)
                    server.stop()

    def test_restarts_same_address_after_http_requests(self):
        server = Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}", models=("mock",))
        for _ in range(2):
            with server:
                self.assertEqual(server.client.list_models().data["models"][0]["name"], "mock")
            self.assertFalse(server.running)

    def test_reuse_probe_rejects_active_listener(self):
        address = f"127.0.0.1:{free_port()}"
        with Server(binary=str(FIXTURE), http_addr=address, models=("mock",)) as owned:
            other = Server(binary=str(FIXTURE), http_addr=address, models=("mock",))
            with self.assertRaises(ServerError):
                other.start()
            self.assertFalse(other.running)
            self.assertTrue(owned.running)
            self.assertEqual(owned.client.list_models().data["models"][0]["name"], "mock")

    def test_does_not_inherit_api_keys(self):
        aliases = ("OPENKIND_API_KEY", "OPENDECISION_API_KEY", "OPENPICK_API_KEY", "TYPESAFE_API_KEY")
        with mock.patch.dict(os.environ, {name: "inherited-secret" for name in aliases}):
            with mock.patch("openkind_client.server.subprocess.Popen", wraps=subprocess.Popen) as spawn:
                with Server(binary=str(FIXTURE), http_addr=f"127.0.0.1:{free_port()}", models=("mock",)) as server:
                    self.assertIsNone(server.api_key)
                    self.assertEqual(server.client.api_key, "")
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
            with mock.patch("openkind_client.server._OPENER.open", side_effect=KeyboardInterrupt):
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
                        http_addr=f"127.0.0.1:{free_port()}", models=("mock",), api_key=generate_api_key())
        for _ in range(2):
            with server:
                self.assertEqual(server.client.health().data["status"], "ok")
                result = server.client.system_one("billing ticket", {
                    "billing": {"type": "noul", "instructions": "Is this billing?"},
                }, model="mock")
                self.assertEqual(result.data["answers"]["billing"]["type"], "noul")
            self.assertFalse(server.running)

    @unittest.skipUnless(os.getenv("OPENKIND_TEST_BINARY"), "set OPENKIND_TEST_BINARY for real openkindd")
    def test_real_binary_without_api_key(self):
        aliases = ("OPENKIND_API_KEY", "OPENDECISION_API_KEY", "OPENPICK_API_KEY", "TYPESAFE_API_KEY")
        with mock.patch.dict(os.environ, {name: "inherited-secret" for name in aliases}):
            with Server(binary=os.environ["OPENKIND_TEST_BINARY"],
                        http_addr=f"127.0.0.1:{free_port()}", models=("mock",)) as server:
                self.assertIsNone(server.api_key)
                self.assertEqual(server.client.api_key, "")
                self.assertEqual(server.client.list_models().data["models"][0]["name"], "mock")


if __name__ == "__main__":
    unittest.main()
