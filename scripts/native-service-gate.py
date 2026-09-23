#!/usr/bin/env python3
"""Run an attributed, queue-inclusive native HTTP service campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import platform
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_ID = "a047d6802c3f06f085b8"
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_requests(fixture: Path, model_alias: str) -> tuple[list[dict[str, Any]], str]:
    raw = fixture.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(stable_json(row["state"]), []).append(row)

    requests = []
    for group in groups.values():
        questions: dict[str, dict[str, Any]] = {}
        for row in group:
            spec = row["primitive"]
            question: dict[str, Any] = {
                "type": spec,
                "instructions": row.get("text", "Decide."),
            }
            if spec == "noul":
                if row.get("criteria") is not None:
                    question["criteria"] = row["criteria"]
            elif spec == "choice":
                criteria = {
                    option["id"]: option.get("description")
                    for option in row["options"]
                }
                criteria.setdefault(
                    "__none__", "None of the listed options applies."
                )
                question["criteria"] = criteria
            elif spec == "score":
                question["criteria"] = row["levels"]
            else:
                raise ValueError(f"unsupported fixture primitive: {spec}")
            questions[row["id"]] = question
        requests.append(
            {
                "state": group[0]["state"],
                "model": model_alias,
                "questions": questions,
            }
        )
    return requests, hashlib.sha256(raw).hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize(samples: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    accepted = [sample for sample in samples if sample["status"] == 200]
    latencies = [sample["latency_ms"] for sample in accepted]
    answered = sum(sample.get("answer_count", 0) for sample in accepted)
    statuses: dict[str, int] = {}
    for sample in samples:
        key = str(sample["status"])
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "attempts": len(samples),
        "http_200": sum(sample["status"] == 200 for sample in samples),
        "http_422": sum(sample["status"] == 422 for sample in samples),
        "http_504": sum(sample["status"] == 504 for sample in samples),
        "http_529": sum(sample["status"] == 529 for sample in samples),
        "transport_errors": sum(sample["status"] == 0 for sample in samples),
        "http_status_counts": statuses,
        "accepted_questions": answered,
        "wall_seconds": wall_seconds,
        "accepted_requests_per_second": len(accepted) / wall_seconds
        if wall_seconds > 0
        else 0.0,
        "accepted_questions_per_second": answered / wall_seconds
        if wall_seconds > 0
        else 0.0,
        "accepted_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "samples": len(latencies),
        },
        "request_id_echoes": sum(
            sample.get("request_id_echoed", False) for sample in samples
        ),
    }


def http_request(
    base_url: str,
    path: str,
    api_key: str,
    request_id: str,
    body: bytes | None = None,
    timeout_seconds: float = 650.0,
) -> tuple[int, dict[str, Any] | None, bool, float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-typesafe-request-id": request_id,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}", data=body, headers=headers, method="POST" if body else "GET"
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
            parsed = json.loads(payload) if path.endswith("systemone") else None
            echoed = response.headers.get("x-typesafe-request-id") == request_id
            return response.status, parsed, echoed, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as error:
        try:
            error.read()
        finally:
            echoed = error.headers.get("x-typesafe-request-id") == request_id
        return error.code, None, echoed, (time.monotonic() - started) * 1000
    except Exception:
        return 0, None, False, (time.monotonic() - started) * 1000


def request_sample(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    request_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    status, response, echoed, latency_ms = http_request(
        base_url,
        "/v1/systemone",
        api_key,
        request_id,
        body,
        timeout_seconds,
    )
    return {
        "status": status,
        "latency_ms": latency_ms,
        "question_count": len(payload["questions"]),
        "answer_count": len(response.get("answers", {})) if response else 0,
        "request_id_echoed": echoed,
    }


def run_deadline_probe(
    command: list[str],
    repo: Path,
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    timeout_ms = 1
    deadline_command = list(command)
    timeout_index = deadline_command.index("--qwen35-timeout-ms") + 1
    deadline_command[timeout_index] = str(timeout_ms)
    deadline_port = free_port()
    deadline_host = "127.0.0.1"
    deadline_url = f"http://{deadline_host}:{deadline_port}"
    deadline_command[deadline_command.index("--http-addr") + 1] = (
        f"{deadline_host}:{deadline_port}"
    )
    deadline_key = secrets.token_urlsafe(32)
    deadline_env = os.environ.copy()
    deadline_env["OPENKIND_API_KEY"] = deadline_key
    deadline_env["RUST_LOG"] = "info"
    log_path = output_dir / "deadline-daemon.log"
    proc: subprocess.Popen[Any] | None = None
    startup_seconds: float | None = None
    sample: dict[str, Any] | None = None
    error_type: str | None = None
    try:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                deadline_command,
                cwd=repo,
                env=deadline_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            startup_seconds = wait_ready(proc, deadline_url, deadline_key)
            sample = request_sample(
                deadline_url,
                deadline_key,
                payload,
                "native-gate-deadline",
                30.0,
            )
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    except Exception as error:
        error_type = type(error).__name__
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    clean_shutdown = proc is not None and proc.returncode == 0
    passed = (
        sample is not None
        and sample["status"] == 504
        and sample["request_id_echoed"]
        and clean_shutdown
    )
    return {
        "configured_timeout_ms": timeout_ms,
        "startup_seconds_to_health": startup_seconds,
        "sample": sample,
        "shutdown_clean": clean_shutdown,
        "error_type": error_type,
        "passed": passed,
    }


def gate_checks(report: dict[str, Any]) -> dict[str, bool]:
    phases = report["phases"]
    initial = all(
        phases[name]["status"] == 200
        and phases[name]["answer_count"] == phases[name]["question_count"]
        and phases[name]["request_id_echoed"]
        for name in ("first_request", "resident_request")
    )
    validation = (
        phases["validation"]["status"] == 422
        and phases["validation"]["malformed_json_status"] == 400
    )
    concurrency = all(
        phases[f"concurrency_{count}"]["http_200"] > 0
        and phases[f"concurrency_{count}"]["http_200"]
        + phases[f"concurrency_{count}"]["http_529"]
        == phases[f"concurrency_{count}"]["attempts"]
        for count in (1, 2, 4)
    ) and phases["concurrency_4"]["http_529"] > 0
    recovery = (
        phases["cancellation_recovery"]["cancellation_observed"]
        and phases["cancellation_recovery"]["recovery"]["status"] == 200
        and phases["cancellation_recovery"]["recovery"]["request_id_echoed"]
    )
    soak = (
        phases["soak"]["attempts"] > 0
        and phases["soak"]["http_200"] == phases["soak"]["attempts"]
        and phases["soak"]["request_id_echoes"] == phases["soak"]["attempts"]
    )
    metrics = (
        phases["metrics"]["process_peak_rss_bytes"] is not None
        and phases["metrics"]["queue_wait_seconds_sum"] is not None
        and phases["metrics"]["queue_wait_seconds_sum"] > 0
        and phases["metrics"]["execution_seconds_sum"] is not None
        and phases["metrics"]["request_seconds_sum"] is not None
    )
    load = phases["concurrency_4"]
    latency_throughput = (
        load["accepted_latency_ms"]["p50"] is not None
        and load["accepted_latency_ms"]["p95"] is not None
        and load["accepted_latency_ms"]["p99"] is not None
        and load["accepted_requests_per_second"] > 0
        and load["accepted_questions_per_second"] > 0
    )
    memory = (
        report["sampled_process_rss"]["samples"] > 0
        and report["sampled_process_rss"]["max_bytes"] is not None
    )
    shutdown = report["shutdown"]["clean"]
    deadline = report["deadline_probe"]["passed"]
    release_artifact = (
        report["daemon_artifact"]["build_profile"] == "release"
        and bool(report["daemon_artifact"]["sha256"])
    )
    source = report["source"]
    source_provenance = (
        source["build_matches_run_start"]
        and report["daemon_artifact"]["unchanged_during_run"]
    )
    return {
        "release_daemon_artifact": release_artifact,
        "source_to_artifact_provenance": source_provenance,
        "first_and_resident_requests": initial,
        "wire_validation": validation,
        "bounded_concurrency_and_overload": concurrency,
        "queue_inclusive_latency_and_throughput": latency_throughput,
        "cancellation_and_recovery": recovery,
        "queue_inclusive_telemetry": metrics,
        "process_memory_sampling": memory,
        "steady_state_soak": soak,
        "queue_inclusive_deadline": deadline,
        "clean_shutdown": shutdown,
    }


def process_rss_bytes(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        kib = int(result.stdout.strip())
        return kib * 1024
    except (ValueError, subprocess.CalledProcessError):
        return None


class RssSampler:
    def __init__(self, pid: int, interval_seconds: float = 1.0) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.values: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = process_rss_bytes(self.pid)
            if value is not None:
                self.values.append((time.monotonic(), value))
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        values = [value for _, value in self.values]
        return {
            "samples": len(values),
            "interval_seconds": self.interval_seconds,
            "min_bytes": min(values) if values else None,
            "median_bytes": percentile([float(value) for value in values], 0.50),
            "max_bytes": max(values) if values else None,
            "first_bytes": values[0] if values else None,
            "last_bytes": values[-1] if values else None,
        }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def metric_value(base_url: str, metric_name: str) -> float | None:
    request = urllib.request.Request(f"{base_url}/metrics", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            for line in response.read().decode("utf-8", errors="replace").splitlines():
                if line.startswith(metric_name + " "):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
    return None


def wait_ready(proc: subprocess.Popen[Any], base_url: str, api_key: str) -> float:
    started = time.monotonic()
    while time.monotonic() - started < 1800:
        if proc.poll() is not None:
            raise RuntimeError(f"openkindd exited during startup with code {proc.returncode}")
        status, _, _, _ = http_request(
            base_url, "/health", api_key, "native-gate-health", timeout_seconds=3
        )
        if status == 200:
            return time.monotonic() - started
        time.sleep(0.25)
    raise TimeoutError("openkindd did not become healthy within 30 minutes")


def disconnect_during_request(
    host: str,
    port: int,
    api_key: str,
    request_id: str,
    payload: dict[str, Any],
    delay_seconds: float,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = (
        f"POST /v1/systemone HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Authorization: Bearer {api_key}\r\n"
        f"x-typesafe-request-id: {request_id}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii")
    with socket.create_connection((host, port), timeout=10) as conn:
        conn.sendall(headers + body)
        time.sleep(delay_seconds)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def machine_record() -> dict[str, Any]:
    def sysctl(name: str) -> str | None:
        try:
            return subprocess.check_output(
                ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    memory = sysctl("hw.memsize")
    return {
        "label": f"{sysctl('hw.model') or platform.node()} {sysctl('machdep.cpu.brand_string') or platform.processor()}",
        "architecture": platform.machine(),
        "os": platform.platform(),
        "memory_bytes": int(memory) if memory and memory.isdigit() else None,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    }


def git_record(repo: Path, harness_path: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "crates", ".cargo"],
        cwd=repo,
        text=True,
    )
    untracked_digest = hashlib.sha256()
    for name in sorted(untracked.splitlines()):
        path = repo / name
        if not path.is_file():
            continue
        untracked_digest.update(name.encode("utf-8"))
        untracked_digest.update(b"\0")
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                untracked_digest.update(chunk)
    return {
        "commit": commit,
        "dirty_tree": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_crate_source_sha256": untracked_digest.hexdigest(),
        "harness_sha256": sha256_file(harness_path),
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--capture-source":
        repo = Path(sys.argv[2]).resolve()
        print(json.dumps(git_record(repo, Path(__file__).resolve()), indent=2))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--daemon", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue", type=int, default=2)
    parser.add_argument("--timeout-ms", type=int, default=600_000)
    parser.add_argument("--max-process-bytes", type=int, default=28 * 1024**3)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--soak-seconds", type=float, default=1800)
    parser.add_argument("--disconnect-after-ms", type=int, default=250)
    parser.add_argument("--request-timeout-seconds", type=float, default=650)
    parser.add_argument("--build-command", type=str, default=None)
    parser.add_argument("--build-source-record", type=Path, required=True)
    parser.add_argument(
        "--build-profile", choices=("release", "debug", "custom"), default=None
    )
    args = parser.parse_args()
    build_profile = args.build_profile or (
        args.daemon.parent.name
        if args.daemon.parent.name in {"release", "debug"}
        else "custom"
    )

    if args.queue < 0 or args.waves < 1 or args.soak_seconds < 0:
        parser.error("queue and waves must be valid; soak-seconds cannot be negative")
    for path in (
        args.repo,
        args.daemon,
        args.bundle_root,
        args.runtime_manifest,
        args.checkpoint_root,
        args.tokenizer,
        args.fixture,
        args.build_source_record,
    ):
        if not path.exists():
            parser.error(f"required local input does not exist: {path}")

    try:
        build_source = json.loads(args.build_source_record.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read build source record: {exc}")
    source_at_start = git_record(args.repo, Path(__file__).resolve())
    source_identity_fields = (
        "commit",
        "tracked_diff_sha256",
        "untracked_crate_source_sha256",
        "harness_sha256",
    )
    build_matches_run_start = all(
        build_source.get(field) == source_at_start[field]
        for field in source_identity_fields
    )
    if not build_matches_run_start:
        parser.error("source changed between daemon build and service run")
    daemon_sha256_at_start = sha256_file(args.daemon)

    model_alias = "qwen35-native"
    payloads, fixture_sha256 = load_requests(args.fixture, model_alias)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "daemon.log"
    report_path = args.output_dir / "service-report.json"
    api_key = secrets.token_urlsafe(32)
    port = free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    env = os.environ.copy()
    env["OPENKIND_API_KEY"] = api_key
    env["RUST_LOG"] = "info"
    command = [
        str(args.daemon),
        "--http-addr",
        f"{host}:{port}",
        "--grpc-addr",
        "0",
        "--models",
        model_alias,
        "--qwen35-aliases",
        model_alias,
        "--qwen35-bundle-root",
        str(args.bundle_root),
        "--qwen35-checkpoint-root",
        str(args.checkpoint_root),
        "--qwen35-tokenizer",
        str(args.tokenizer),
        "--qwen35-concurrency",
        "1",
        "--qwen35-queue",
        str(args.queue),
        "--qwen35-timeout-ms",
        str(args.timeout_ms),
        "--rate-limit-rpm",
        "0",
        "--qwen35-max-process-bytes",
        str(args.max_process_bytes),
    ]

    samples: list[dict[str, Any]] = []
    phases: dict[str, Any] = {}
    cancelled_before = 0.0
    cancelled_after = 0.0
    proc: subprocess.Popen[Any] | None = None
    sampler: RssSampler | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] | None = None
    try:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            startup_seconds = wait_ready(proc, base_url, api_key)
            sampler = RssSampler(proc.pid)
            sampler.start()

            for phase, payload in (
                ("first_request", payloads[0]),
                ("resident_request", payloads[1 % len(payloads)]),
            ):
                sample = request_sample(
                    base_url,
                    api_key,
                    payload,
                    f"native-gate-{phase}",
                    args.request_timeout_seconds,
                )
                samples.append({"phase": phase, **sample})
                phases[phase] = sample

            invalid_body = json.dumps(
                {"state": {}, "model": model_alias, "questions": {}}
            ).encode("utf-8")
            invalid_status, _, invalid_id_echoed, invalid_ms = http_request(
                base_url,
                "/v1/systemone",
                api_key,
                "native-gate-invalid-wire",
                invalid_body,
                args.request_timeout_seconds,
            )
            bad_json_status, _, bad_json_id_echoed, bad_json_ms = http_request(
                base_url,
                "/v1/systemone",
                api_key,
                "native-gate-invalid-json",
                b"{invalid-json",
                args.request_timeout_seconds,
            )
            phases["validation"] = {
                "status": invalid_status,
                "latency_ms": invalid_ms,
                "request_id_echoed": invalid_id_echoed,
                "malformed_json_status": bad_json_status,
                "malformed_json_latency_ms": bad_json_ms,
                "malformed_json_request_id_echoed": bad_json_id_echoed,
            }

            for concurrency in (1, 2, 4):
                phase_samples: list[dict[str, Any]] = []
                phase_started = time.monotonic()
                for wave in range(args.waves):
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=concurrency
                    ) as pool:
                        futures = [
                            pool.submit(
                                request_sample,
                                base_url,
                                api_key,
                                payloads[(wave * concurrency + index) % len(payloads)],
                                f"native-gate-c{concurrency}-w{wave}-r{index}",
                                args.request_timeout_seconds,
                            )
                            for index in range(concurrency)
                        ]
                        phase_samples.extend(future.result() for future in futures)
                wall = time.monotonic() - phase_started
                samples.extend(
                    {"phase": f"concurrency_{concurrency}", **sample}
                    for sample in phase_samples
                )
                phases[f"concurrency_{concurrency}"] = summarize(phase_samples, wall)

            cancelled_before = metric_value(
                base_url, "openkind_native_cancellations_total"
            ) or 0.0
            disconnect_during_request(
                host,
                port,
                api_key,
                "native-gate-disconnect",
                payloads[0],
                args.disconnect_after_ms / 1000.0,
            )
            cancel_wait_started = time.monotonic()
            while time.monotonic() - cancel_wait_started < 15:
                cancelled_after = metric_value(
                    base_url, "openkind_native_cancellations_total"
                ) or 0.0
                if cancelled_after > cancelled_before:
                    break
                time.sleep(0.1)
            recovery_started = time.monotonic()
            recovery = request_sample(
                base_url,
                api_key,
                payloads[1 % len(payloads)],
                "native-gate-recovery",
                args.request_timeout_seconds,
            )
            recovery_seconds = time.monotonic() - recovery_started
            phases["cancellation_recovery"] = {
                "disconnect_after_ms": args.disconnect_after_ms,
                "cancellation_metric_before": cancelled_before,
                "cancellation_metric_after": cancelled_after,
                "cancellation_observed": cancelled_after > cancelled_before,
                "recovery": recovery,
                "recovery_wall_seconds": recovery_seconds,
            }
            samples.append({"phase": "post_cancellation_recovery", **recovery})

            soak_samples: list[dict[str, Any]] = []
            soak_started = time.monotonic()
            soak_deadline = soak_started + args.soak_seconds
            index = 0
            while time.monotonic() < soak_deadline:
                sample = request_sample(
                    base_url,
                    api_key,
                    payloads[index % len(payloads)],
                    f"native-gate-soak-{index}",
                    args.request_timeout_seconds,
                )
                soak_samples.append(sample)
                index += 1
                if index % 5 == 0:
                    time.sleep(0.05)
            soak_wall = time.monotonic() - soak_started
            phases["soak"] = summarize(soak_samples, soak_wall)
            samples.extend({"phase": "soak", **sample} for sample in soak_samples)

            peak_rss = metric_value(
                base_url, "openkind_native_process_peak_rss_bytes"
            )
            phases["metrics"] = {
                "process_peak_rss_bytes": int(peak_rss) if peak_rss else None,
                "queue_wait_seconds_sum": metric_value(
                    base_url, "openkind_native_queue_wait_seconds_sum"
                ),
                "execution_seconds_sum": metric_value(
                    base_url, "openkind_native_execution_seconds_sum"
                ),
                "request_seconds_sum": metric_value(
                    base_url, "openkind_native_request_seconds_sum"
                ),
            }

            report = {
                "schema": "openkind-native-service-gate/v1",
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "machine": machine_record(),
                "source": {
                    **source_at_start,
                    "build_source_record": build_source,
                    "build_matches_run_start": build_matches_run_start,
                },
                "daemon_artifact": {
                    "path": os.path.relpath(args.daemon.resolve(), args.repo.resolve()),
                    "sha256": daemon_sha256_at_start,
                    "build_profile": build_profile,
                    "build_command": (
                        args.build_command
                        or (
                            "cargo build --release --locked -p openkind-server --bin openkindd"
                            if build_profile == "release"
                            else None
                        )
                    ),
                },
                "profile": {
                    "profile_id": PROFILE_ID,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "bundle_manifest_sha256": sha256_file(
                        args.bundle_root / "BUNDLE_MANIFEST.json"
                    ),
                    "bundle_runtime_sha256": sha256_file(args.runtime_manifest),
                    "checkpoint_index_sha256": sha256_file(
                        args.checkpoint_root / "model.safetensors.index.json"
                    ),
                    "fixture_sha256": fixture_sha256,
                    "fixture_requests": len(payloads),
                    "fixture_questions": sum(
                        len(payload["questions"]) for payload in payloads
                    ),
                    "accepted_error_metrics": "unavailable: fixture has no reviewed labels",
                },
                "configuration": {
                    "concurrency": 1,
                    "queue": args.queue,
                    "timeout_ms": args.timeout_ms,
                    "max_process_bytes": args.max_process_bytes,
                    "soak_seconds": args.soak_seconds,
                    "payloads_saved": False,
                    "request_ids_saved": False,
                    "api_key_saved": False,
                },
                "startup_seconds_to_health": startup_seconds,
                "phases": phases,
                "samples": samples,
                "sampled_process_rss": sampler.summary() if sampler else None,
            }
    finally:
        if sampler:
            sampler.stop()
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    if report is not None:
        source_at_finish = git_record(args.repo, Path(__file__).resolve())
        daemon_sha256_at_finish = sha256_file(args.daemon)
        report["source"].update(
            {
                "tracked_source_unchanged_during_run": (
                    source_at_start["commit"] == source_at_finish["commit"]
                    and source_at_start["tracked_diff_sha256"]
                    == source_at_finish["tracked_diff_sha256"]
                    and source_at_start["untracked_crate_source_sha256"]
                    == source_at_finish["untracked_crate_source_sha256"]
                ),
                "harness_unchanged_during_run": (
                    source_at_start["harness_sha256"]
                    == source_at_finish["harness_sha256"]
                ),
                "finish_tracked_diff_sha256": source_at_finish[
                    "tracked_diff_sha256"
                ],
                "finish_commit": source_at_finish["commit"],
                "finish_untracked_crate_source_sha256": source_at_finish[
                    "untracked_crate_source_sha256"
                ],
            }
        )
        report["daemon_artifact"].update(
            {
                "finish_sha256": daemon_sha256_at_finish,
                "unchanged_during_run": daemon_sha256_at_start
                == daemon_sha256_at_finish,
            }
        )
        report["shutdown"] = {
            "signal": "SIGINT",
            "exit_code": proc.returncode if proc else None,
            "clean": proc is not None and proc.returncode == 0,
        }
        report["deadline_probe"] = run_deadline_probe(
            command,
            args.repo,
            payloads[0],
            args.output_dir,
        )
        checks = gate_checks(report)
        report["gate"] = {"passed": all(checks.values()), "checks": checks}
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    if proc is not None and proc.returncode != 0:
        print(f"openkindd exited with {proc.returncode}; see {log_path}", file=sys.stderr)
        return 1
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
