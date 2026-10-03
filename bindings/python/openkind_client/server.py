"""Lifecycle wrapper for an existing openkindd executable."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import time
from collections.abc import Sequence
from urllib.error import URLError
from urllib.request import urlopen

from . import Client


class ServerError(RuntimeError):
    pass


class Server:
    """Start and stop one local openkindd child process."""

    def __init__(
        self,
        binary: str = "openkindd",
        http_addr: str = "127.0.0.1:18080",
        models: Sequence[str] = ("mock", "jev-latest"),
        api_key: str | None = None,
        startup_timeout: float = 10.0,
        shutdown_timeout: float = 5.0,
        extra_args: Sequence[str] = (),
    ) -> None:
        try:
            host, port_text = http_addr.rsplit(":", 1)
            port = int(port_text)
        except ValueError as error:
            raise ValueError("http_addr must be a loopback host:port") from error
        if host != "127.0.0.1" or not 1 <= port <= 65535:
            raise ValueError("http_addr must use 127.0.0.1 and a nonzero port")
        if not models or any(not model or "," in model for model in models):
            raise ValueError("models must contain nonempty aliases without commas")
        if (not math.isfinite(startup_timeout) or not math.isfinite(shutdown_timeout)
                or startup_timeout <= 0 or shutdown_timeout <= 0):
            raise ValueError("timeouts must be positive")
        self.binary = binary
        self.http_addr = http_addr
        self.base_url = f"http://{http_addr}"
        self.models = tuple(models)
        self.api_key = api_key
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.extra_args = tuple(extra_args)
        managed_flags = ("--http-addr", "--grpc-addr", "--models", "--api-key")
        if any(arg == flag or arg.startswith(flag + "=")
               for arg in self.extra_args for flag in managed_flags):
            raise ValueError("extra_args cannot override managed server flags")
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def client(self) -> Client:
        return Client(base_url=self.base_url, api_key=self.api_key if self.api_key is not None else "")

    def start(self) -> Server:
        if self._process is not None:
            raise ServerError("server has already been started")
        host, port_text = self.http_addr.rsplit(":", 1)
        with socket.socket() as probe:
            try:
                probe.bind((host, int(port_text)))
            except OSError as error:
                raise ServerError(f"HTTP address is unavailable: {self.http_addr}") from error
        args = [self.binary, *self.extra_args, "--http-addr", self.http_addr,
                "--grpc-addr", "0", "--models", ",".join(self.models)]
        child_env = os.environ.copy()
        child_env.pop("OPENKIND_API_KEY", None)
        child_env.pop("OPENDECISION_API_KEY", None)
        child_env.pop("OPENPICK_API_KEY", None)
        child_env.pop("TYPESAFE_API_KEY", None)
        if self.api_key is not None:
            child_env["OPENKIND_API_KEY"] = self.api_key
        try:
            self._process = subprocess.Popen(args, env=child_env)
        except OSError as error:
            raise ServerError(f"could not start openkindd: {error}") from error
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    code = self._process.returncode
                    self._process = None
                    raise ServerError(f"openkindd exited before readiness with code {code}")
                try:
                    with urlopen(self.base_url + "/health", timeout=0.25) as response:
                        if response.status == 200 and json.load(response).get("status") == "ok":
                            return self
                except (OSError, URLError, ValueError, AttributeError):
                    pass
                time.sleep(0.05)
            raise ServerError("openkindd did not become healthy before the startup timeout")
        except BaseException:
            # __exit__ is not called when context-manager entry fails or is interrupted.
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
        self._process = None

    def __enter__(self) -> Server:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
