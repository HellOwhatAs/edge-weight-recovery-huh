#!/usr/bin/env python3
"""Crash-safe, receipt-driven runner for the route-baseline study.

The runner deliberately has no experiment-specific imports.  A declarative JSON
file defines ordered tasks, immutable prerequisites, expected outputs and the
progress heartbeat produced by a long-running adapter.  Successful tasks get an
atomic receipt that binds the exact command, inputs and outputs.  A later run
skips a task only after re-hashing and validating all of that evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


CONFIG_SCHEMA = "ewr.route-baseline-pipeline/v1"
STATE_SCHEMA = "ewr.route-baseline-pipeline-state/v1"
RECEIPT_SCHEMA = "ewr.route-baseline-task-receipt/v1"
LAUNCH_SCHEMA = "ewr.route-baseline-pipeline-launch/v1"
TASK_TIME_EVIDENCE_SCHEMA = "ewr.route-baseline-task-time-evidence/v1"
RUNNER_VERSION = "1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "route_baselines_full_test_20260719"
    / "pipeline.json"
)
TOKEN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PipelineError(RuntimeError):
    """An expected, user-readable pipeline failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def read_json(path: Path, context: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineError(f"{context} is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {context} {path}: {error}") from error


def nested_value(value: Any, dotted_key: str) -> Any:
    current = value
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise PipelineError(f"JSON field {dotted_key!r} is missing")
        current = current[component]
    return current


class Pipeline:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        config = read_json(self.config_path, "pipeline configuration")
        if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
            raise PipelineError(
                f"pipeline configuration must use schema {CONFIG_SCHEMA}"
            )
        self.config: dict[str, Any] = config
        configured_root = config.get("workspace_root", "../../..")
        root = Path(str(configured_root))
        if not root.is_absolute():
            root = self.config_path.parent / root
        self.workspace = root.resolve()
        if not self.workspace.is_dir():
            raise PipelineError(f"workspace root is not a directory: {self.workspace}")
        self.runner_path = Path(__file__).resolve()
        self.runner_sha256 = sha256_file(self.runner_path)

        raw_variables = config.get("variables", {})
        if not isinstance(raw_variables, dict):
            raise PipelineError("variables must be an object")
        self.variables: dict[str, str] = {
            "workspace": str(self.workspace),
            **{str(key): str(value) for key, value in raw_variables.items()},
        }
        runtime = self.expand(str(config.get("runtime_root", "research/generated")))
        runtime_path = Path(runtime)
        if not runtime_path.is_absolute():
            runtime_path = self.workspace / runtime_path
        self.runtime_root = runtime_path.resolve()
        self.variables["runtime"] = str(self.runtime_root)

        state_dir = self.expand(str(config.get("state_dir", ".pipeline")))
        state_path = Path(state_dir)
        if not state_path.is_absolute():
            state_path = self.runtime_root / state_path
        self.state_dir = state_path.resolve()
        self.state_path = self.state_dir / "state.json"
        self.launch_path = self.state_dir / "launch.json"
        self.receipt_dir = self.state_dir / "receipts"
        self.log_dir = self.state_dir / "logs"
        self.lock_path = self.state_dir / "runner.lock"

        tasks = config.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise PipelineError("pipeline tasks must be a non-empty array")
        self.tasks: list[dict[str, Any]] = []
        self.task_by_id: dict[str, dict[str, Any]] = {}
        for raw_task in tasks:
            if not isinstance(raw_task, dict):
                raise PipelineError("every task must be an object")
            task = dict(raw_task)
            task_id = task.get("id")
            if not isinstance(task_id, str) or not re.fullmatch(
                r"[a-z][a-z0-9_-]*", task_id
            ):
                raise PipelineError(f"invalid task id: {task_id!r}")
            if task_id in self.task_by_id:
                raise PipelineError(f"duplicate task id: {task_id}")
            action = task.get("action", "command")
            if action not in {"command", "verify"}:
                raise PipelineError(f"task {task_id}: unsupported action {action!r}")
            profiles = task.get("profiles", ["full"])
            if not isinstance(profiles, list) or not profiles or any(
                profile not in {"smoke", "full"} for profile in profiles
            ):
                raise PipelineError(f"task {task_id}: invalid profiles")
            if action == "command":
                command = task.get("command")
                if (
                    not isinstance(command, list)
                    or not command
                    or any(not isinstance(part, str) or not part for part in command)
                ):
                    raise PipelineError(f"task {task_id}: command must be a string array")
                self._validate_device_policy(task)
            for field in ("requires", "outputs"):
                specs = task.get(field, [])
                if not isinstance(specs, list) or any(
                    not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
                    for spec in specs
                ):
                    raise PipelineError(f"task {task_id}: {field} must contain file specs")
            bindings = task.get("hash_bindings", [])
            if not isinstance(bindings, list) or any(
                not isinstance(binding, dict)
                or set(binding) != {"json_path", "field", "file_path"}
                or not all(isinstance(value, str) for value in binding.values())
                for binding in bindings
            ):
                raise PipelineError(
                    f"task {task_id}: hash_bindings must contain exact JSON/file bindings"
                )
            self.tasks.append(task)
            self.task_by_id[task_id] = task

        seen: set[str] = set()
        for task in self.tasks:
            for dependency in task.get("depends_on", []):
                if dependency not in self.task_by_id:
                    raise PipelineError(
                        f"task {task['id']}: unknown dependency {dependency!r}"
                    )
                if dependency not in seen:
                    raise PipelineError(
                        f"task {task['id']}: dependency {dependency!r} must appear earlier"
                    )
            seen.add(task["id"])

    def _validate_device_policy(self, task: Mapping[str, Any]) -> None:
        device = task.get("device", "cpu")
        if device not in {"cpu", "cuda:0", "mixed:cuda:0+cpu"}:
            raise PipelineError(f"task {task['id']}: invalid resolved device {device!r}")
        if device in {"cuda:0", "mixed:cuda:0+cpu"}:
            command = task["command"]
            try:
                index = command.index("--device")
                argument = command[index + 1]
            except (ValueError, IndexError) as error:
                raise PipelineError(
                    f"task {task['id']}: CUDA task must explicitly pass --device"
                ) from error
            if argument not in {"cuda", "cuda:0"}:
                raise PipelineError(
                    f"task {task['id']}: CUDA task may not use device {argument!r}"
                )

    def expand(self, value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self.variables:
                raise PipelineError(f"unknown pipeline variable {name!r}")
            return self.variables[name]

        previous = value
        for _ in range(10):
            expanded = TOKEN.sub(replace, previous)
            if expanded == previous:
                return expanded
            previous = expanded
        raise PipelineError(f"variable expansion is recursive: {value!r}")

    def path(self, value: str) -> Path:
        expanded = Path(self.expand(value))
        if not expanded.is_absolute():
            expanded = self.workspace / expanded
        return expanded.resolve()

    def selected_tasks(self, profile: str) -> list[dict[str, Any]]:
        included_profiles = {"smoke"} if profile == "smoke" else {"smoke", "full"}
        selected = [
            task
            for task in self.tasks
            if included_profiles.intersection(task.get("profiles", ["full"]))
        ]
        selected_ids = {task["id"] for task in selected}
        for task in selected:
            missing = set(task.get("depends_on", [])) - selected_ids
            if missing:
                raise PipelineError(
                    f"task {task['id']}: selected profile omits dependencies {sorted(missing)}"
                )
        return selected

    def fingerprint(self, spec: Mapping[str, Any], context: str) -> dict[str, Any]:
        path = self.path(str(spec["path"]))
        if not path.is_file():
            raise PipelineError(f"{context} is missing or not a file: {path}")
        size = path.stat().st_size
        if "bytes" in spec and size != int(spec["bytes"]):
            raise PipelineError(
                f"{context} size mismatch for {path}: expected {spec['bytes']}, got {size}"
            )
        if size < int(spec.get("min_bytes", 1)):
            raise PipelineError(f"{context} is unexpectedly small ({size} bytes): {path}")
        digest = sha256_file(path)
        expected = spec.get("sha256")
        if expected is not None and digest != expected:
            raise PipelineError(
                f"{context} SHA-256 mismatch for {path}: expected {expected}, got {digest}"
            )
        json_contains = spec.get("json_contains")
        if json_contains is not None:
            if not isinstance(json_contains, dict):
                raise PipelineError(f"{context}: json_contains must be an object")
            value = read_json(path, context)
            for dotted_key, wanted in json_contains.items():
                try:
                    actual = nested_value(value, str(dotted_key))
                except PipelineError as error:
                    raise PipelineError(f"{context} {path}: {error}") from error
                if actual != wanted:
                    raise PipelineError(
                        f"{context} {path}: field {dotted_key!r} expected "
                        f"{wanted!r}, got {actual!r}"
                    )
        return {"path": str(path), "bytes": size, "sha256": digest}

    def fingerprints(
        self, specs: Iterable[Mapping[str, Any]], context: str
    ) -> list[dict[str, Any]]:
        return [
            self.fingerprint(spec, f"{context} #{index + 1}")
            for index, spec in enumerate(specs)
        ]

    def validate_hash_bindings(self, task: Mapping[str, Any]) -> None:
        for index, binding in enumerate(task.get("hash_bindings", [])):
            json_path = self.path(binding["json_path"])
            file_path = self.path(binding["file_path"])
            document = read_json(
                json_path, f"task {task['id']} hash binding #{index + 1} JSON"
            )
            recorded = nested_value(document, binding["field"])
            if not isinstance(recorded, str):
                raise PipelineError(
                    f"task {task['id']} hash binding field {binding['field']!r} "
                    "must be a SHA-256 string"
                )
            actual = sha256_file(file_path)
            if recorded != actual:
                raise PipelineError(
                    f"task {task['id']} hash binding mismatch: "
                    f"{json_path}:{binding['field']} records {recorded}, "
                    f"but {file_path} hashes to {actual}"
                )

    def expanded_command(self, task: Mapping[str, Any]) -> list[str]:
        return [self.expand(str(part)) for part in task["command"]]

    def task_identity(self, task: Mapping[str, Any], execution_mode: str) -> str:
        action = task.get("action", "command")
        allowed_modes = (
            {"verification_only"}
            if action == "verify"
            else {"direct", "systemd_scope"}
        )
        if execution_mode not in allowed_modes:
            raise PipelineError(
                f"task {task['id']}: invalid receipt execution mode {execution_mode!r}"
            )
        relevant = {
            key: task[key]
            for key in sorted(task)
            if key != "description"
        }
        if task.get("action", "command") == "command":
            relevant["resolved_command"] = self.expanded_command(task)
        return canonical_sha256(
            {
                "runner_version": RUNNER_VERSION,
                "runner_sha256": self.runner_sha256,
                "task": relevant,
                "execution_defaults": {
                    "cpu_threads": self.config.get("cpu_threads", 16),
                    "environment": self.config.get("environment", {}),
                    "resources": self.config.get("resources", {}),
                    "configured_systemd_scope": task.get("systemd_scope", True),
                    "receipt_execution_mode": execution_mode,
                    "cuda_visible_devices": (
                        "0"
                        if task.get("device") in {"cuda:0", "mixed:cuda:0+cpu"}
                        else None
                    ),
                },
            }
        )

    def receipt_path(self, task_id: str) -> Path:
        return self.receipt_dir / f"{task_id}.json"

    def validate_receipt(
        self,
        task: Mapping[str, Any],
        *,
        expected_direct: bool | None = None,
        raise_on_error: bool = False,
    ) -> tuple[bool, str]:
        if task.get("always_run", False):
            return False, "configured to run every time"
        receipt_path = self.receipt_path(str(task["id"]))
        try:
            receipt = read_json(receipt_path, "task receipt")
            if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
                raise PipelineError("unsupported receipt schema")
            if receipt.get("task_id") != task["id"]:
                raise PipelineError("receipt task identity mismatch")
            execution_mode = receipt.get("execution_mode")
            expected_mode = (
                None
                if expected_direct is None
                else ("direct" if expected_direct else "systemd_scope")
            )
            if expected_mode is not None and execution_mode != expected_mode:
                raise PipelineError(
                    f"receipt execution mode is {execution_mode!r}, expected {expected_mode!r}"
                )
            if receipt.get("task_identity_sha256") != self.task_identity(
                task, str(execution_mode)
            ):
                raise PipelineError("task command/configuration changed")
            requires = self.fingerprints(
                task.get("requires", []), f"task {task['id']} prerequisite"
            )
            outputs = self.fingerprints(
                task.get("outputs", []), f"task {task['id']} output"
            )
            self.validate_hash_bindings(task)
            if receipt.get("requires") != requires:
                raise PipelineError("prerequisite fingerprints changed")
            if receipt.get("outputs") != outputs:
                raise PipelineError("output fingerprints changed")
            return True, "receipt, prerequisites and outputs verified"
        except PipelineError as error:
            if raise_on_error:
                raise
            return False, str(error)

    def write_receipt(
        self,
        task: Mapping[str, Any],
        requires: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        *,
        disposition: str,
        attempt: int,
        time_report: str | None,
        execution_mode: str,
    ) -> dict[str, Any]:
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "runner_version": RUNNER_VERSION,
            "runner_path": str(self.runner_path),
            "runner_sha256": self.runner_sha256,
            "task_id": task["id"],
            "task_identity_sha256": self.task_identity(task, execution_mode),
            "execution_mode": execution_mode,
            "disposition": disposition,
            "attempt": attempt,
            "device": task.get("device", "cpu"),
            "execution_note": task.get("execution_note"),
            "requires": requires,
            "outputs": outputs,
            "time_report": time_report,
            "completed_at": utc_now(),
        }
        atomic_write_json(self.receipt_path(str(task["id"])), receipt)
        return receipt

    def read_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        value = read_json(self.state_path, "pipeline state")
        if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
            raise PipelineError(f"unsupported pipeline state: {self.state_path}")
        return value

    def write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def _read_progress(self, task: Mapping[str, Any]) -> dict[str, Any] | None:
        value = task.get("progress_path")
        if not value:
            return None
        path = self.path(str(value))
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(progress, dict):
            return None
        try:
            modified = dt.datetime.fromtimestamp(
                path.stat().st_mtime, tz=dt.timezone.utc
            ).isoformat(timespec="seconds")
        except OSError:
            modified = None
        if modified is not None:
            # This is runner-owned display metadata, not a mutation of the
            # adapter's versioned progress document.
            progress["progress_file_updated_at"] = modified
        return progress

    def _new_state(self, profile: str) -> dict[str, Any]:
        old = self.read_state() or {}
        old_tasks = old.get("tasks", {}) if isinstance(old.get("tasks"), dict) else {}
        tasks: dict[str, Any] = {}
        for task in self.selected_tasks(profile):
            previous = old_tasks.get(task["id"], {})
            attempts = previous.get("attempts", []) if isinstance(previous, dict) else []
            tasks[task["id"]] = {
                "status": "pending",
                "device": task.get("device", "cpu"),
                "execution_note": task.get("execution_note"),
                "attempts": attempts if isinstance(attempts, list) else [],
            }
        return {
            "schema": STATE_SCHEMA,
            "runner_version": RUNNER_VERSION,
            "run_id": f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
            "profile": profile,
            "status": "running",
            "workspace": str(self.workspace),
            "config": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "launcher_unit": os.environ.get("EWR_PIPELINE_LAUNCHER_UNIT"),
            "started_at": utc_now(),
            "current_task": None,
            "tasks": tasks,
        }

    def _time_path(self, task_id: str, attempt: int) -> Path:
        return self.log_dir / f"{task_id}.attempt-{attempt}.time.txt"

    def _log_path(self, task_id: str, attempt: int) -> Path:
        return self.log_dir / f"{task_id}.attempt-{attempt}.log"

    def _scope_name(self, task_id: str, attempt: int) -> str:
        return f"ewr-route-{task_id}-{os.getpid()}-a{attempt}"

    def _write_time_aggregate(
        self, task: Mapping[str, Any], task_state: Mapping[str, Any]
    ) -> None:
        raw_output = task.get("time_aggregate_output")
        if raw_output is None:
            return
        attempts = task_state.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise PipelineError(
                f"task {task['id']}: cannot aggregate missing timing attempts"
            )
        evidence: list[dict[str, Any]] = []
        lost_attempts: list[int] = []
        for index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, Mapping):
                raise PipelineError(f"task {task['id']}: malformed attempt #{index}")
            raw_path = attempt.get("time_report")
            if not raw_path:
                raise PipelineError(
                    f"task {task['id']}: attempt #{index} has no time report"
                )
            report_path = Path(str(raw_path)).resolve()
            try:
                report = report_path.read_text(encoding="utf-8")
            except OSError:
                report = ""
            recorded_status = attempt.get("status")
            exit_match = re.search(r"(?m)^\s*Exit status:\s*(-?\d+)\s*$", report)
            exit_status = int(exit_match.group(1)) if exit_match else None
            required_time_labels = (
                "User time (seconds):",
                "System time (seconds):",
                "Elapsed (wall clock) time",
                "Maximum resident set size (kbytes):",
                "Exit status:",
            )
            timing_complete = (
                bool(report.strip())
                and exit_status is not None
                and all(label in report for label in required_time_labels)
            )
            if timing_complete:
                resolved_status = "succeeded" if exit_status == 0 else "failed"
                if recorded_status in {"succeeded", "failed"} and recorded_status != resolved_status:
                    raise PipelineError(
                        f"task {task['id']}: attempt #{index} state/time exit status differ"
                    )
            else:
                resolved_status = "lost"
                lost_attempts.append(index)
            encoded_report = report.encode("utf-8")
            evidence.append(
                {
                    "attempt": int(attempt.get("attempt", index)),
                    "recorded_status": recorded_status,
                    "status": resolved_status,
                    "timing_status": "complete" if timing_complete else "lost",
                    "return_code": (
                        attempt.get("return_code")
                        if attempt.get("return_code") is not None
                        else exit_status
                    ),
                    "started_at": attempt.get("started_at"),
                    "ended_at": attempt.get("ended_at"),
                    "time_report_path": str(report_path),
                    "time_report_bytes": len(encoded_report) if timing_complete else None,
                    "time_report_sha256": (
                        hashlib.sha256(encoded_report).hexdigest()
                        if timing_complete
                        else None
                    ),
                    "time_report_text": report if timing_complete else None,
                }
            )
        if evidence[-1]["status"] not in {"succeeded", "lost"}:
            raise PipelineError(
                f"task {task['id']}: final timing attempt did not succeed"
            )
        output = self.path(str(raw_output))
        atomic_write_json(
            output,
            {
                "schema": TASK_TIME_EVIDENCE_SCHEMA,
                "task_id": task["id"],
                "attempt_count": len(evidence),
                "attempts": evidence,
                "timing_complete": not lost_attempts,
                "lost_attempts": lost_attempts,
                "aggregation_rule": (
                    "sum elapsed wall seconds across every active attempt; "
                    "exclude downtime between attempts; if timing_complete is false, "
                    "known attempt wall is only a lower bound"
                ),
                "created_at": utc_now(),
            },
        )

    def _run_command(
        self,
        task: Mapping[str, Any],
        state: dict[str, Any],
        *,
        direct: bool,
        poll_seconds: float,
    ) -> tuple[int, str, str, str | None]:
        task_state = state["tasks"][task["id"]]
        attempt = len(task_state["attempts"]) + 1
        log_path = self._log_path(str(task["id"]), attempt)
        time_path = self._time_path(str(task["id"]), attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        base_command = self.expanded_command(task)
        timed_command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_path),
            *base_command,
        ]
        scope_name: str | None = None
        command = timed_command
        if not direct and task.get("systemd_scope", True):
            scope_name = self._scope_name(str(task["id"]), attempt)
            command = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                f"--unit={scope_name}",
            ]
            memory = self.config.get("resources", {})
            for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "OOMPolicy"):
                if key in memory:
                    command.append(f"--property={key}={memory[key]}")
            command.extend(timed_command)

        env = os.environ.copy()
        threads = str(self.config.get("cpu_threads", 16))
        env.update(
            {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "RAYON_NUM_THREADS": threads,
            }
        )
        configured_env = self.config.get("environment", {})
        task_env = task.get("environment", {})
        for source in (configured_env, task_env):
            if not isinstance(source, dict):
                raise PipelineError("pipeline/task environment must be an object")
            env.update({str(key): self.expand(str(value)) for key, value in source.items()})
        if task.get("device") in {"cuda:0", "mixed:cuda:0+cpu"}:
            env["CUDA_VISIBLE_DEVICES"] = "0"

        attempt_state = {
            "attempt": attempt,
            "status": "running",
            "started_at": utc_now(),
            "log": str(log_path),
            "time_report": str(time_path),
            "scope_unit": f"{scope_name}.scope" if scope_name else None,
            "command": base_command,
        }
        task_state["attempts"].append(attempt_state)
        task_state.update(
            {
                "status": "running",
                "started_at": attempt_state["started_at"],
                "log": str(log_path),
                "time_report": str(time_path),
                "scope_unit": attempt_state["scope_unit"],
            }
        )
        state["current_task"] = task["id"]
        self.write_state(state)

        stop_requested = False
        previous_handlers: dict[int, Any] = {}

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True
            attempt_state["stop_signal"] = signal.Signals(signum).name
            self.write_state(state)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

        try:
            with log_path.open("ab", buffering=0) as log:
                header = (
                    f"\n[{utc_now()}] attempt={attempt} device={task_state['device']}\n"
                    f"command={json.dumps(base_command, ensure_ascii=False)}\n"
                ).encode("utf-8")
                log.write(header)
                process = subprocess.Popen(
                    command,
                    cwd=self.workspace,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt_state["launcher_pid"] = process.pid
                task_state["launcher_pid"] = process.pid
                self.write_state(state)
                while True:
                    return_code = process.poll()
                    progress = self._read_progress(task)
                    if progress is not None:
                        task_state["progress"] = progress
                    self.write_state(state)
                    if return_code is not None:
                        break
                    if stop_requested:
                        if scope_name:
                            subprocess.run(
                                ["systemctl", "--user", "stop", f"{scope_name}.scope"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                            )
                        else:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        try:
                            return_code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            return_code = process.wait()
                        break
                    time.sleep(poll_seconds)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        ended_at = utc_now()
        attempt_state.update(
            {
                "status": "succeeded" if return_code == 0 else "failed",
                "return_code": return_code,
                "ended_at": ended_at,
            }
        )
        task_state["ended_at"] = ended_at
        task_state["return_code"] = return_code
        task_state["status"] = "succeeded" if return_code == 0 else "failed"
        self.write_state(state)
        return return_code, str(log_path), str(time_path), scope_name

    def run(self, profile: str, *, direct: bool, poll_seconds: float) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock_path.open("a+b")
        try:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PipelineError(
                    f"another pipeline runner holds {self.lock_path}; use status instead"
                ) from error
            state = self._new_state(profile)
            self.write_state(state)
            for task in self.selected_tasks(profile):
                task_id = str(task["id"])
                task_state = state["tasks"][task_id]
                action = task.get("action", "command")
                valid, reason = self.validate_receipt(
                    task,
                    expected_direct=(direct if action == "command" else None),
                )
                if valid:
                    task_state.update(
                        {"status": "skipped_verified", "reason": reason, "ended_at": utc_now()}
                    )
                    self.write_state(state)
                    continue

                requires = self.fingerprints(
                    task.get("requires", []), f"task {task_id} prerequisite"
                )
                if task.get("action", "command") == "verify":
                    outputs = self.fingerprints(
                        task.get("outputs", []), f"task {task_id} immutable output"
                    )
                    self.validate_hash_bindings(task)
                    self.write_receipt(
                        task,
                        requires,
                        outputs,
                        disposition="verified_existing",
                        attempt=0,
                        time_report=None,
                        execution_mode="verification_only",
                    )
                    task_state.update(
                        {"status": "verified_existing", "reason": reason, "ended_at": utc_now()}
                    )
                    self.write_state(state)
                    continue

                if task.get("adopt_outputs", False):
                    # A command may have exited successfully just before the
                    # runner was interrupted while promoting its stable GNU
                    # time report. Recover that exact prior attempt before
                    # validating adoptable outputs; never replace it with a
                    # near-zero resume-only measurement.
                    if "time_output" in task:
                        formal_time = self.path(str(task["time_output"]))
                        if not formal_time.is_file():
                            for prior in reversed(task_state.get("attempts", [])):
                                candidate = prior.get("time_report")
                                if prior.get("status") in {"succeeded", "running"} and candidate:
                                    candidate_path = Path(str(candidate))
                                    if candidate_path.is_file():
                                        atomic_write_bytes(
                                            formal_time, candidate_path.read_bytes()
                                        )
                                        break
                    if task_state.get("attempts"):
                        self._write_time_aggregate(task, task_state)
                    try:
                        outputs = self.fingerprints(
                            task.get("outputs", []), f"task {task_id} adoptable output"
                        )
                    except PipelineError:
                        outputs = []
                    if outputs and len(outputs) == len(task.get("outputs", [])):
                        self.validate_hash_bindings(task)
                        adopted_time: str | None = None
                        adopted_attempt = 0
                        for prior in reversed(task_state.get("attempts", [])):
                            candidate = prior.get("time_report")
                            if prior.get("status") == "succeeded" and candidate:
                                candidate_path = Path(candidate)
                                if candidate_path.is_file():
                                    adopted_time = str(candidate_path)
                                    adopted_attempt = int(prior.get("attempt", 0))
                                    if "time_output" in task:
                                        atomic_write_bytes(
                                            self.path(str(task["time_output"])),
                                            candidate_path.read_bytes(),
                                        )
                                    break
                        self.write_receipt(
                            task,
                            requires,
                            outputs,
                            disposition="adopted_validated_outputs",
                            attempt=adopted_attempt,
                            time_report=adopted_time,
                            execution_mode="direct" if direct else "systemd_scope",
                        )
                        task_state.update(
                            {"status": "adopted_validated", "ended_at": utc_now()}
                        )
                        self.write_state(state)
                        continue

                if task.get("immutable", False) and any(
                    self.path(spec["path"]).exists() for spec in task.get("outputs", [])
                ):
                    raise PipelineError(
                        f"task {task_id} has invalid immutable outputs; refusing to overwrite"
                    )
                for spec in task.get("outputs", []):
                    self.path(spec["path"]).parent.mkdir(parents=True, exist_ok=True)
                return_code, log_path, time_path, _ = self._run_command(
                    task, state, direct=direct, poll_seconds=poll_seconds
                )
                if return_code != 0:
                    raise PipelineError(
                        f"task {task_id} failed with exit code {return_code}; log: {log_path}"
                    )
                if "time_output" in task:
                    formal_time = self.path(str(task["time_output"]))
                    atomic_write_bytes(formal_time, Path(time_path).read_bytes())
                self._write_time_aggregate(task, task_state)
                outputs = self.fingerprints(
                    task.get("outputs", []), f"task {task_id} output"
                )
                self.validate_hash_bindings(task)
                attempt = len(task_state["attempts"])
                self.write_receipt(
                    task,
                    requires,
                    outputs,
                    disposition="executed",
                    attempt=attempt,
                    time_report=time_path,
                    execution_mode="direct" if direct else "systemd_scope",
                )
                task_state.update(
                    {"status": "succeeded", "receipt": str(self.receipt_path(task_id))}
                )
                state["current_task"] = None
                self.write_state(state)

            state["status"] = "succeeded"
            state["current_task"] = None
            state["ended_at"] = utc_now()
            self.write_state(state)
        except BaseException as error:
            try:
                state
            except UnboundLocalError:
                raise
            state["status"] = "failed"
            state["error"] = str(error)
            state["ended_at"] = utc_now()
            self.write_state(state)
            raise
        finally:
            lock_handle.close()

    def launch(self, profile: str, unit: str | None) -> str:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        unit_name = unit or str(
            self.config.get("launcher_unit", f"ewr-route-baselines-{profile}")
        )
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", unit_name):
            raise PipelineError(f"invalid systemd unit name: {unit_name!r}")
        command = [
            "systemd-run",
            "--user",
            f"--unit={unit_name}",
            "--collect",
            "--same-dir",
            f"--setenv=EWR_PIPELINE_LAUNCHER_UNIT={unit_name}.service",
        ]
        resources = self.config.get("resources", {})
        for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "OOMPolicy"):
            if key in resources:
                command.append(f"--property={key}={resources[key]}")
        command.extend(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(self.config_path),
                "run",
                "--profile",
                profile,
            ]
        )
        launch = {
            "schema": LAUNCH_SCHEMA,
            "profile": profile,
            "unit": f"{unit_name}.service",
            "command": command,
            "requested_at": utc_now(),
            "status": "launching",
        }
        atomic_write_json(self.launch_path, launch)
        result = subprocess.run(command, cwd=self.workspace, text=True, capture_output=True)
        if result.returncode != 0:
            launch.update(
                {"status": "failed", "error": (result.stderr or result.stdout).strip()}
            )
            atomic_write_json(self.launch_path, launch)
            raise PipelineError(f"could not launch pipeline: {launch['error']}")
        launch.update({"status": "launched", "launcher_output": result.stdout.strip()})
        atomic_write_json(self.launch_path, launch)
        return f"{unit_name}.service"

    def stop(self) -> list[str]:
        units: list[str] = []
        state = self.read_state()
        if state:
            current = state.get("current_task")
            task_state = state.get("tasks", {}).get(current, {}) if current else {}
            scope = task_state.get("scope_unit") if isinstance(task_state, dict) else None
            if scope:
                units.append(str(scope))
            launcher = state.get("launcher_unit")
            if launcher:
                units.append(str(launcher))
        if self.launch_path.exists():
            launch = read_json(self.launch_path, "launch state")
            if isinstance(launch, dict) and launch.get("unit"):
                units.append(str(launch["unit"]))
        stopped: list[str] = []
        for unit in dict.fromkeys(units):
            subprocess.run(
                ["systemctl", "--user", "stop", unit],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            stopped.append(unit)
        return stopped

    def status(self, *, verify: bool, as_json: bool) -> None:
        state = self.read_state()
        if state is None:
            snapshot: dict[str, Any] = {
                "schema": STATE_SCHEMA,
                "status": "not_started",
                "state_path": str(self.state_path),
            }
        else:
            snapshot = json.loads(json.dumps(state))
            current = snapshot.get("current_task")
            if current and current in self.task_by_id:
                progress = self._read_progress(self.task_by_id[current])
                if progress is not None:
                    snapshot["tasks"][current]["progress"] = progress
            if verify:
                # Do not even stat/hash a future task's inputs.  This is essential
                # for keeping the frozen formal test split unopened until model
                # selection (the training task) has a verified receipt.
                verified_dependencies: set[str] = set()
                for task in self.selected_tasks(snapshot.get("profile", "full")):
                    dependencies = set(task.get("depends_on", []))
                    if not dependencies.issubset(verified_dependencies):
                        snapshot["tasks"][task["id"]]["verification"] = (
                            "deferred_until_dependencies_complete"
                        )
                        continue
                    if task.get("always_run", False):
                        successful = snapshot["tasks"][task["id"]].get("status") in {
                            "succeeded",
                            "skipped_verified",
                            "verified_existing",
                            "adopted_validated",
                        }
                        snapshot["tasks"][task["id"]]["verification"] = (
                            "completed_this_run"
                            if successful
                            else "must_complete_in_current_run"
                        )
                        if successful:
                            verified_dependencies.add(task["id"])
                        continue
                    valid, reason = self.validate_receipt(task)
                    snapshot["tasks"][task["id"]]["verification"] = (
                        "verified" if valid else reason
                    )
                    if valid:
                        verified_dependencies.add(task["id"])
        if as_json:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
            return
        print(f"pipeline: {snapshot.get('status')}")
        print(f"state:    {self.state_path}")
        if snapshot.get("profile"):
            print(f"profile:  {snapshot['profile']}")
        if snapshot.get("started_at"):
            print(f"started:  {snapshot['started_at']}")
        if snapshot.get("error"):
            print(f"error:    {snapshot['error']}")
        current = snapshot.get("current_task")
        if current:
            task_state = snapshot["tasks"][current]
            print(f"current:  {current}")
            print(f"device:   {task_state.get('device', 'unknown')}")
            if task_state.get("execution_note"):
                print(f"compute:  {task_state['execution_note']}")
            if task_state.get("started_at"):
                elapsed = seconds_since(task_state["started_at"])
                if elapsed is not None:
                    print(f"elapsed:  {format_duration(elapsed)}")
            progress = task_state.get("progress")
            if isinstance(progress, dict):
                phase = first_present(progress, ("stage", "phase", "status"))
                if phase:
                    print(f"phase:    {phase}")
                completed = first_present(
                    progress, ("completed_units", "completed_samples")
                )
                total = first_present(progress, ("total_units", "total_samples"))
                unit = progress.get("unit")
                if unit is None and (
                    "completed_samples" in progress or "total_samples" in progress
                ):
                    unit = "samples"
                if completed is not None and total is not None:
                    print(f"progress: {completed}/{total} {unit or 'units'}")
                else:
                    percent = first_present(progress, ("phase_percent", "percent"))
                    if percent is not None:
                        print(f"progress: {percent}%")
                recoverable = first_present(
                    progress,
                    ("recoverable_completed_units", "completed_samples"),
                )
                if recoverable is not None:
                    print(
                        f"durable:  {recoverable}/{total if total is not None else '?'} "
                        f"{unit or 'units'} recoverable"
                    )
                maximum_redo = progress.get("maximum_redo_units")
                if maximum_redo is not None:
                    print(f"redo max: {maximum_redo} {unit or 'units'} after interruption")
                pipeline_percent = progress.get("pipeline_percent")
                if isinstance(pipeline_percent, (int, float)):
                    print(f"overall:  {pipeline_percent:.2f}% (three-stage estimate)")
                eta = first_present(
                    progress,
                    (
                        "phase_eta_seconds",
                        "eta_seconds",
                        "estimated_remaining_adapter_seconds",
                        "estimated_remaining_prediction_seconds",
                    ),
                )
                if (
                    eta is None
                    and isinstance(completed, (int, float))
                    and isinstance(total, (int, float))
                    and completed > 0
                    and total >= completed
                ):
                    measured = first_present(
                        progress,
                        ("prediction_seconds", "completed_adapter_seconds"),
                    )
                    if isinstance(measured, (int, float)):
                        eta = float(measured) * (float(total) - float(completed)) / float(completed)
                if isinstance(eta, (int, float)):
                    print(f"ETA:      {format_duration(float(eta))}")
                heartbeat = first_present(
                    progress,
                    (
                        "updated_at", "updated_at_utc", "last_heartbeat",
                        "progress_file_updated_at",
                    ),
                )
                if heartbeat:
                    age = seconds_since(str(heartbeat))
                    suffix = f" ({format_duration(age)} ago)" if age is not None else ""
                    print(f"heartbeat:{' ' if suffix else '  '}{heartbeat}{suffix}")
            if task_state.get("log"):
                print(f"log:      {task_state['log']}")
        print("tasks:")
        for task_id, task_state in snapshot.get("tasks", {}).items():
            detail = task_state.get("verification")
            suffix = f" — {detail}" if detail else ""
            print(f"  {task_id:32} {task_state.get('status', 'unknown')}{suffix}")

    def show_log(self, task_id: str | None, lines: int) -> None:
        state = self.read_state()
        if state is None:
            raise PipelineError("pipeline has not started")
        selected = task_id or state.get("current_task")
        if not selected:
            attempts = [
                (key, value)
                for key, value in state.get("tasks", {}).items()
                if value.get("log")
            ]
            if not attempts:
                raise PipelineError("no task log is available")
            selected = attempts[-1][0]
        task_state = state.get("tasks", {}).get(selected)
        if not isinstance(task_state, dict) or not task_state.get("log"):
            raise PipelineError(f"no log is recorded for task {selected!r}")
        path = Path(task_state["log"])
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            raise PipelineError(f"cannot read task log {path}: {error}") from error
        print("\n".join(content[-lines:]))


def first_present(value: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if value.get(key) is not None:
            return value[key]
    return None


def parse_timestamp(value: str) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def seconds_since(value: str) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds())


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate configuration without opening data")
    run = commands.add_parser("run", help="run in the foreground with resumable receipts")
    run.add_argument("--profile", choices=["smoke", "full"], required=True)
    run.add_argument(
        "--direct",
        action="store_true",
        help="do not create per-task systemd scopes (intended only for tests)",
    )
    run.add_argument("--poll-seconds", type=float, default=5.0)
    launch = commands.add_parser(
        "launch", help="launch a durable detached systemd user service"
    )
    launch.add_argument("--profile", choices=["smoke", "full"], required=True)
    launch.add_argument("--unit")
    status = commands.add_parser("status", help="show progress without touching future data")
    status.add_argument("--verify", action="store_true")
    status.add_argument("--json", action="store_true")
    log = commands.add_parser("log", help="show the tail of a task log")
    log.add_argument("--task")
    log.add_argument("--lines", type=int, default=80)
    commands.add_parser("stop", help="gracefully stop the active task/service")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = Pipeline(args.config)
        if args.command == "validate":
            print(f"valid: {pipeline.config_path}")
        elif args.command == "run":
            if args.poll_seconds <= 0:
                raise PipelineError("--poll-seconds must be positive")
            pipeline.run(
                args.profile, direct=args.direct, poll_seconds=args.poll_seconds
            )
            print(f"pipeline {args.profile} completed successfully")
        elif args.command == "launch":
            unit = pipeline.launch(args.profile, args.unit)
            print(f"launched {unit}; Codex/terminal may now be closed")
            print(
                f"status: {Path(__file__).resolve()} --config {pipeline.config_path} status"
            )
        elif args.command == "status":
            pipeline.status(verify=args.verify, as_json=args.json)
        elif args.command == "log":
            if args.lines <= 0:
                raise PipelineError("--lines must be positive")
            pipeline.show_log(args.task, args.lines)
        else:
            stopped = pipeline.stop()
            print("stopped: " + (", ".join(stopped) if stopped else "no active unit found"))
        return 0
    except PipelineError as error:
        print(f"pipeline error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
