"""Resumable, progress-reporting trainer for the DRNCS-LG adapter.

This module is deliberately separate from :mod:`drncs_lg_adapter`.  In
particular, adding recovery support must not change the adapter source hash
that is bound into an expensive, already completed preprocessing artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import drncs_lg_adapter as adapter


RUNNER_VERSION = "0.1.0"
PROGRESS_SCHEMA = "ewr.drncs-lg-training-progress/v1"
RESUME_SCHEMA = "ewr.drncs-lg-training-resume/v1"
PROVENANCE_SCHEMA = "ewr.drncs-lg-resumable-training-provenance/v1"
RUNNER_NAME = "drncs_lg_resumable"

_STOP_SIGNAL: int | None = None


class StopRequested(RuntimeError):
    """Raised only at a recovery-safe boundary after SIGINT or SIGTERM."""


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum


def _raise_if_stop_requested() -> None:
    if _STOP_SIGNAL is not None:
        name = signal.Signals(_STOP_SIGNAL).name
        raise StopRequested(f"stop requested by {name}")


def reset_stop_request_for_tests() -> None:
    """Clear process-local signal state; intended for in-process tests only."""

    global _STOP_SIGNAL
    _STOP_SIGNAL = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runner_source_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "name": RUNNER_NAME,
        "version": RUNNER_VERSION,
        "path": str(path),
        "sha256": adapter.sha256_file(path),
    }


def _resource_peaks(previous: Any = None) -> dict[str, int]:
    prior = previous if isinstance(previous, dict) else {}
    cuda_peak = 0
    if adapter.torch is not None and adapter.torch.cuda.is_available():
        cuda_peak = int(adapter.torch.cuda.max_memory_allocated())
    return {
        "peak_rss_kib": max(int(prior.get("peak_rss_kib", 0)), adapter.peak_rss_kib()),
        "peak_cuda_memory_bytes": max(
            int(prior.get("peak_cuda_memory_bytes", 0)), cuda_peak
        ),
    }


class ProgressReporter:
    """Write a small atomic heartbeat without making progress outrun recovery."""

    def __init__(
        self,
        path: Path,
        *,
        binding_sha256: str,
        source: dict[str, str],
        runner: dict[str, str],
        device: str,
        output_dir: Path,
        prior_process_seconds: float,
        invocation_started: float,
    ) -> None:
        self.path = path
        self.binding_sha256 = binding_sha256
        self.source = source
        self.runner = runner
        self.device = device
        self.output_dir = output_dir
        self.prior_process_seconds = max(0.0, prior_process_seconds)
        self.invocation_started = invocation_started
        self.last: dict[str, Any] | None = None
        self._checkpoint_cache: dict[Path, tuple[int, int, str]] = {}
        self._last_write_monotonic = 0.0

    def heartbeat_due(self, interval_seconds: float = 2.0) -> bool:
        return time.perf_counter() - self._last_write_monotonic >= interval_seconds

    def write(
        self,
        *,
        status: str,
        stage: str,
        stage_index: int,
        completed_units: int,
        total_units: int,
        unit: str,
        recoverable_completed_units: int | None = None,
        phase_elapsed_seconds: float,
        retained_compute_seconds: float,
        checkpoint: Path | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        completed = max(0, int(completed_units))
        total = max(0, int(total_units))
        recoverable = (
            completed
            if recoverable_completed_units is None
            else max(0, int(recoverable_completed_units))
        )
        if recoverable > completed:
            raise RuntimeError("recoverable progress cannot exceed live progress")
        phase_fraction = 1.0 if total == 0 else min(1.0, completed / total)
        phase_eta = None
        if 0 < completed < total and phase_elapsed_seconds > 0:
            phase_eta = phase_elapsed_seconds * (total - completed) / completed
        pipeline_fraction = min(1.0, (stage_index + phase_fraction) / 3.0)
        checkpoint_value = None
        if checkpoint is not None and checkpoint.is_file():
            stat = checkpoint.stat()
            cached = self._checkpoint_cache.get(checkpoint)
            if cached is not None and cached[:2] == (stat.st_mtime_ns, stat.st_size):
                checkpoint_hash = cached[2]
            else:
                checkpoint_hash = adapter.sha256_file(checkpoint)
                self._checkpoint_cache[checkpoint] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                    checkpoint_hash,
                )
            checkpoint_value = {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
            }
        value: dict[str, Any] = {
            "schema": PROGRESS_SCHEMA,
            "status": status,
            "stage": stage,
            "stage_index": stage_index,
            "stage_count": 3,
            "unit": unit,
            "completed_units": completed,
            "recoverable_completed_units": recoverable,
            "maximum_redo_units": completed - recoverable,
            "total_units": total,
            "phase_fraction": phase_fraction,
            "phase_percent": 100.0 * phase_fraction,
            "pipeline_fraction": pipeline_fraction,
            "pipeline_percent": 100.0 * pipeline_fraction,
            "phase_elapsed_seconds": float(phase_elapsed_seconds),
            "phase_eta_seconds": phase_eta,
            "eta_scope": "current_stage_only",
            "current_run_wall_seconds": now - self.invocation_started,
            "cumulative_process_wall_seconds": (
                self.prior_process_seconds + now - self.invocation_started
            ),
            "cumulative_retained_compute_seconds": float(retained_compute_seconds),
            "device": self.device,
            "output_dir": str(self.output_dir),
            "resume_binding_sha256": self.binding_sha256,
            "adapter_source": self.source,
            "runner": self.runner,
            "checkpoint": checkpoint_value,
            "stop_signal": (
                signal.Signals(_STOP_SIGNAL).name if _STOP_SIGNAL is not None else None
            ),
            "updated_at": utc_now(),
        }
        if message is not None:
            value["message"] = message
        adapter.write_json_atomic(self.path, value)
        self.last = value
        self._last_write_monotonic = time.perf_counter()
        return value


def _load_resume(path: Path, binding_sha256: str, stage: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = adapter.load_torch_artifact(path)
    if not isinstance(value, dict) or value.get("schema") != RESUME_SCHEMA:
        raise RuntimeError(f"unsupported resumable state {path}")
    if value.get("stage") != stage:
        raise RuntimeError(f"resumable state {path} has the wrong stage")
    if value.get("resume_binding_sha256") != binding_sha256:
        raise RuntimeError(
            f"resumable state {path} belongs to different code, inputs, or settings"
        )
    if value.get("runner") != runner_source_identity():
        raise RuntimeError(f"resumable state {path} was produced by a different runner")
    return value


def _prior_process_seconds(progress_path: Path, binding_sha256: str) -> float:
    if not progress_path.is_file():
        return 0.0
    progress = adapter.load_strict_json(progress_path)
    if not isinstance(progress, dict):
        return 0.0
    if progress.get("schema") != PROGRESS_SCHEMA:
        return 0.0
    if progress.get("resume_binding_sha256") != binding_sha256:
        return 0.0
    value = progress.get("cumulative_process_wall_seconds", 0.0)
    return max(0.0, float(value))


def _print_event(stage: str, completed: int, total: int, **extra: Any) -> None:
    value = {
        "event": "drncs_lg_progress",
        "stage": stage,
        "completed": completed,
        "total": total,
        **extra,
    }
    print(json.dumps(value, separators=(",", ":")), flush=True)


def fit_transition_model_resumable(
    model: Any,
    embeddings: Any,
    outgoing: Sequence[Sequence[int]],
    train_trips: Sequence[adapter.Trip],
    validation_predictor: Any,
    validation_truth: Sequence[Sequence[int] | adapter.Trip],
    *,
    stage: str,
    stage_index: int,
    resume_path: Path,
    binding_sha256: str,
    runner: dict[str, str],
    reporter: ProgressReporter,
    retained_before_seconds: float,
    epochs: int,
    validation_every: int,
    batch_size: int,
    transition_chunk_size: int,
    learning_rate: float,
    seed: int,
    device: Any,
) -> tuple[Any, dict[str, Any], float, dict[str, int]]:
    """Fit one MLP, atomically retaining every completed epoch."""

    if (
        epochs <= 0
        or validation_every <= 0
        or transition_chunk_size <= 0
        or learning_rate <= 0
    ):
        raise RuntimeError("epochs, validation interval, and learning rate must be positive")
    if not train_trips or not validation_truth:
        raise RuntimeError("training and validation routes must be non-empty")

    model.to(device)
    embedding_tensor = adapter.torch.from_numpy(embeddings).to(device)
    optimizer = adapter.torch.optim.Adam(model.parameters(), lr=learning_rate)
    state = _load_resume(resume_path, binding_sha256, stage)
    if state is None:
        completed_epoch = 0
        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        best_rank: tuple[float, float, int] | None = None
        total_updates = 0
        phase_wall_seconds = 0.0
        peaks = _resource_peaks()
    else:
        completed_epoch = int(state["completed_epoch"])
        if not 0 <= completed_epoch <= epochs:
            raise RuntimeError(f"invalid completed epoch in {resume_path}")
        model.load_state_dict(state["model_state"])
        model.to(device)
        optimizer.load_state_dict(state["optimizer_state"])
        history = list(state["history"])
        best_state = state["best_state"]
        best_epoch = int(state["best_epoch"])
        raw_rank = state["best_rank"]
        best_rank = tuple(raw_rank) if raw_rank is not None else None
        total_updates = int(state["optimizer_updates"])
        phase_wall_seconds = float(state["phase_wall_seconds"])
        peaks = _resource_peaks(state.get("resource_peaks"))
        if len(history) != completed_epoch:
            raise RuntimeError(f"history length in {resume_path} is inconsistent")

    def checkpoint_payload() -> dict[str, Any]:
        return {
            "schema": RESUME_SCHEMA,
            "stage": stage,
            "complete": completed_epoch == epochs,
            "resume_binding_sha256": binding_sha256,
            "runner": runner,
            "completed_epoch": completed_epoch,
            "model_state": adapter.state_dict_on_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_rank": list(best_rank) if best_rank is not None else None,
            "optimizer_updates": total_updates,
            "phase_wall_seconds": phase_wall_seconds,
            "history": history,
            "resource_peaks": _resource_peaks(peaks),
            "updated_at": utc_now(),
        }

    if state is None:
        adapter.write_torch_atomic(resume_path, checkpoint_payload())
    retained = retained_before_seconds + phase_wall_seconds
    reporter.write(
        status="running",
        stage=stage,
        stage_index=stage_index,
        completed_units=completed_epoch,
        total_units=epochs,
        unit="epoch",
        phase_elapsed_seconds=phase_wall_seconds,
        retained_compute_seconds=retained,
        checkpoint=resume_path,
    )
    try:
        _raise_if_stop_requested()
        for epoch in range(completed_epoch + 1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            epoch_loss = 0.0
            epoch_transitions = 0
            epoch_routes = 0
            for current, destinations, targets, route_count in adapter.transition_minibatches(
                train_trips, outgoing, batch_size, random.Random(seed + epoch)
            ):
                _raise_if_stop_requested()
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                count = len(current)
                for offset in range(0, count, transition_chunk_size):
                    _raise_if_stop_requested()
                    chunk = slice(offset, offset + transition_chunk_size)
                    logits, _, _ = adapter.candidate_logits(
                        model,
                        embedding_tensor,
                        current[chunk],
                        destinations[chunk],
                        outgoing,
                        device,
                    )
                    target_tensor = adapter.torch.as_tensor(
                        targets[chunk], dtype=adapter.torch.long, device=device
                    )
                    loss_sum = adapter.torch.nn.functional.cross_entropy(
                        logits, target_tensor, reduction="sum"
                    )
                    (loss_sum / route_count).backward()
                    batch_loss += float(loss_sum.detach().cpu())
                optimizer.step()
                epoch_loss += batch_loss
                epoch_transitions += count
                epoch_routes += route_count
                total_updates += 1
            if epoch_transitions == 0 or epoch_routes == 0:
                raise RuntimeError("training routes contain no transitions")
            entry: dict[str, Any] = {
                "epoch": epoch,
                "mean_training_loss": epoch_loss / epoch_routes,
                "release_transition_mean_loss": epoch_loss / epoch_transitions,
                "training_routes": epoch_routes,
                "training_transitions": epoch_transitions,
                "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
            }
            if epoch % validation_every == 0 or epoch == epochs:
                adapter.synchronize_device(device)
                validation_started = time.perf_counter()
                predicted, prediction_stats = validation_predictor(model)
                adapter.synchronize_device(device)
                metrics = adapter.route_metrics(validation_truth, predicted)
                entry["validation"] = metrics
                entry["validation_prediction"] = prediction_stats
                entry["validation_seconds"] = time.perf_counter() - validation_started
                rank = (float(metrics["edge_f1"]), float(metrics["exact_match"]), -epoch)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_epoch = epoch
                    best_state = adapter.state_dict_on_cpu(model)
            epoch_seconds = time.perf_counter() - epoch_started
            entry["epoch_wall_seconds"] = epoch_seconds
            history.append(entry)
            completed_epoch = epoch
            phase_wall_seconds += epoch_seconds
            peaks = _resource_peaks(peaks)
            adapter.write_torch_atomic(resume_path, checkpoint_payload())
            retained = retained_before_seconds + phase_wall_seconds
            reporter.write(
                status="running",
                stage=stage,
                stage_index=stage_index,
                completed_units=completed_epoch,
                total_units=epochs,
                unit="epoch",
                phase_elapsed_seconds=phase_wall_seconds,
                retained_compute_seconds=retained,
                checkpoint=resume_path,
            )
            _print_event(
                stage,
                completed_epoch,
                epochs,
                epoch_seconds=epoch_seconds,
                selected_epoch=best_epoch,
            )
            _raise_if_stop_requested()
    except StopRequested as error:
        reporter.write(
            status="interrupted",
            stage=stage,
            stage_index=stage_index,
            completed_units=completed_epoch,
            total_units=epochs,
            unit="epoch",
            phase_elapsed_seconds=phase_wall_seconds,
            retained_compute_seconds=retained_before_seconds + phase_wall_seconds,
            checkpoint=resume_path,
            message=str(error),
        )
        raise

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    diagnostics = {
        "selected_epoch": best_epoch,
        "selected_rank": list(best_rank) if best_rank is not None else None,
        "optimizer_updates": total_updates,
        "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
        "release_difference": "release_uses_transition_mean_cross_entropy",
        "wall_seconds": phase_wall_seconds,
        "history": history,
        "resumable_epoch_checkpoints": True,
    }
    return model, diagnostics, phase_wall_seconds, _resource_peaks(peaks)


def build_sc2_database_resumable(
    model: Any,
    embeddings: Any,
    graph: adapter.LineGraph,
    contraction: adapter.ContractionResult,
    sc1: dict[tuple[int, int], list[int]],
    device: Any,
    *,
    score_batch_size: int,
    checkpoint_every: int,
    resume_path: Path,
    binding_sha256: str,
    runner: dict[str, str],
    reporter: ProgressReporter,
    retained_before_seconds: float,
) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any], float, dict[str, int]]:
    """Construct SC2 by destination with bounded-loss atomic recovery."""

    if score_batch_size <= 0 or checkpoint_every <= 0:
        raise RuntimeError("SC2 score batch size and checkpoint interval must be positive")
    missing = [pair for pair in contraction.shortcut_pairs if pair not in sc1]
    grouped: dict[int, list[int]] = defaultdict(list)
    for source, destination in missing:
        grouped[destination].append(source)
    destinations = sorted(grouped)
    state = _load_resume(resume_path, binding_sha256, "sc2")
    if state is None:
        completed = 0
        recoverable = 0
        database: dict[tuple[int, int], list[int]] = {}
        phase_wall_seconds = 0.0
        peaks = _resource_peaks()
    else:
        completed = int(state["completed_destinations"])
        recoverable = completed
        if not 0 <= completed <= len(destinations):
            raise RuntimeError(f"invalid SC2 destination progress in {resume_path}")
        database = dict(state["database"])
        phase_wall_seconds = float(state["phase_wall_seconds"])
        peaks = _resource_peaks(state.get("resource_peaks"))
        expected_pairs = {
            (source, destination)
            for destination in destinations[:completed]
            for source in grouped[destination]
        }
        if set(database) != expected_pairs:
            raise RuntimeError(f"SC2 database in {resume_path} is inconsistent")

    def save() -> None:
        payload = {
            "schema": RESUME_SCHEMA,
            "stage": "sc2",
            "complete": completed == len(destinations),
            "resume_binding_sha256": binding_sha256,
            "runner": runner,
            "completed_destinations": completed,
            "total_destinations": len(destinations),
            "database": database,
            "phase_wall_seconds": phase_wall_seconds,
            "resource_peaks": _resource_peaks(peaks),
            "updated_at": utc_now(),
        }
        adapter.write_torch_atomic(resume_path, payload)

    if state is None:
        save()
    reporter.write(
        status="running",
        stage="sc2",
        stage_index=1,
        completed_units=completed,
        recoverable_completed_units=recoverable,
        total_units=len(destinations),
        unit="destination",
        phase_elapsed_seconds=phase_wall_seconds,
        retained_compute_seconds=retained_before_seconds + phase_wall_seconds,
        checkpoint=resume_path,
    )
    model.eval()
    embedding_tensor = adapter.torch.from_numpy(embeddings).to(device)
    try:
        _raise_if_stop_requested()
        for index in range(completed, len(destinations)):
            destination = destinations[index]
            unit_started = time.perf_counter()
            paths = adapter.shortest_model_paths(
                model,
                embedding_tensor,
                graph,
                grouped[destination],
                destination,
                device,
                score_batch_size=score_batch_size,
            )
            for source in sorted(grouped[destination]):
                path = paths[source]
                if path is None:
                    raise RuntimeError(
                        f"SC2 cannot connect shortcut {source}->{destination} "
                        "on the original graph"
                    )
                database[(source, destination)] = path
            adapter.synchronize_device(device)
            phase_wall_seconds += time.perf_counter() - unit_started
            completed = index + 1
            peaks = _resource_peaks(peaks)
            must_save = (
                completed % checkpoint_every == 0
                or completed == len(destinations)
                or _STOP_SIGNAL is not None
            )
            if must_save:
                save()
                recoverable = completed
            if must_save or reporter.heartbeat_due():
                reporter.write(
                    status="running",
                    stage="sc2",
                    stage_index=1,
                    completed_units=completed,
                    recoverable_completed_units=recoverable,
                    total_units=len(destinations),
                    unit="destination",
                    phase_elapsed_seconds=phase_wall_seconds,
                    retained_compute_seconds=retained_before_seconds + phase_wall_seconds,
                    checkpoint=resume_path,
                )
            if must_save:
                _print_event("sc2", completed, len(destinations))
            _raise_if_stop_requested()
    except StopRequested as error:
        save()
        recoverable = completed
        reporter.write(
            status="interrupted",
            stage="sc2",
            stage_index=1,
            completed_units=completed,
            recoverable_completed_units=recoverable,
            total_units=len(destinations),
            unit="destination",
            phase_elapsed_seconds=phase_wall_seconds,
            retained_compute_seconds=retained_before_seconds + phase_wall_seconds,
            checkpoint=resume_path,
            message=str(error),
        )
        raise
    save()
    return (
        database,
        {
            "shortcut_pairs": len(database),
            "destinations": len(destinations),
            "wall_seconds": phase_wall_seconds,
            "resumable_destination_checkpoints": True,
            "checkpoint_every_destinations": checkpoint_every,
        },
        phase_wall_seconds,
        _resource_peaks(peaks),
    )


def _configuration(args: argparse.Namespace, workers: int, device: Any) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "workers": workers,
        "device": str(device),
        "epochs": args.epochs,
        "validation_every": args.validation_every,
        "batch_size_routes": args.batch_size,
        "transition_chunk_size": args.transition_chunk_size,
        "learning_rate": args.learning_rate,
        "hidden_dimension": args.hidden_dimension,
        "inference_batch_size": args.inference_batch_size,
        "max_steps": args.max_steps,
        "sc2_score_batch_size": args.sc2_score_batch_size,
        "sc2_checkpoint_every_destinations": args.sc2_checkpoint_every,
        "checkpoint_selection": "validation_macro_edge_f1_then_exact_match_then_earliest_epoch",
        "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
        "release_loss_difference": "release_cross_entropy_uses_transition_mean",
        "train_split_role_enforced": True,
        "validation_split_role_enforced": True,
        "train_manifest_hash_pin_enforced": args.expected_train_manifest_sha256 is not None,
        "train_records_hash_pin_enforced": args.expected_train_records_sha256 is not None,
        "train_dataset_hash_pins_enforced": all(
            value is not None
            for value in (
                args.expected_train_manifest_sha256,
                args.expected_train_records_sha256,
            )
        ),
        "validation_manifest_hash_pin_enforced": (
            args.expected_validation_manifest_sha256 is not None
        ),
        "validation_records_hash_pin_enforced": (
            args.expected_validation_records_sha256 is not None
        ),
        "validation_dataset_hash_pins_enforced": all(
            value is not None
            for value in (
                args.expected_validation_manifest_sha256,
                args.expected_validation_records_sha256,
            )
        ),
        "sparse_training_storage": "lazy_view_with_uint32_base_route_indices",
        "resumable_training": True,
        "epoch_checkpoint_interval": 1,
        "total_process_timing_boundary": (
            "cumulative_active_invocations_through_final_checkpoint_before_diagnostics_write"
        ),
        "phase_timing_boundary": "successfully_retained_complete_units_only",
    }


def train_command(args: argparse.Namespace) -> None:
    invocation_started = time.perf_counter()
    workers = adapter.configure_runtime(args.seed, args.workers)
    adapter.load_array_dependencies()
    adapter.configure_runtime(args.seed, workers)
    device = adapter.device_from_name(args.device)
    if device.type == "cuda":
        adapter.torch.cuda.reset_peak_memory_stats(device)
    runner = runner_source_identity()
    source = adapter.adapter_source_identity(args.source_revision)
    artifact, preprocess_metadata = adapter.load_preprocess_directory(args.preprocess_dir)
    adapter.require_same_adapter_source(artifact.get("source"), "preprocessing artifact")
    if source != artifact.get("source"):
        raise RuntimeError("training source identity differs from preprocessing")
    train_dataset = adapter.load_dataset_manifest(
        args.train_manifest,
        expected_role="train",
        expected_manifest_sha256=args.expected_train_manifest_sha256,
        expected_records_sha256=args.expected_train_records_sha256,
    )
    validation_dataset = adapter.load_dataset_manifest(
        args.validation_manifest,
        expected_role="validation",
        expected_manifest_sha256=args.expected_validation_manifest_sha256,
        expected_records_sha256=args.expected_validation_records_sha256,
    )
    adapter.require_same_dataset(train_dataset, artifact["train_dataset"], "training manifest")
    if validation_dataset.manifest.network_id != train_dataset.manifest.network_id:
        raise RuntimeError("training and validation manifests use different networks")
    adapter.require_disjoint_samples(train_dataset, validation_dataset)

    configuration = _configuration(args, workers, device)
    binding = {
        "schema": RESUME_SCHEMA,
        "runner": runner,
        "adapter_source": source,
        "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
        "train_dataset": adapter.dataset_binding(train_dataset),
        "validation_dataset": adapter.dataset_binding(validation_dataset),
        "configuration": configuration,
    }
    binding_sha256 = adapter.sha256_bytes(adapter.canonical_json_bytes(binding))
    output_dir = args.output_dir.resolve()
    resume_dir = (args.resume_dir or output_dir / "resume").resolve()
    progress_path = (args.progress or output_dir / "progress.json").resolve()
    original_resume = resume_dir / "original.pt"
    sc2_resume = resume_dir / "sc2.pt"
    sparse_resume = resume_dir / "sparse.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    diagnostics_path = output_dir / "training_diagnostics.json"
    provenance_path = output_dir / "resume_provenance.json"
    existing = [
        path
        for path in (
            original_resume,
            sc2_resume,
            sparse_resume,
            checkpoint_path,
            diagnostics_path,
            provenance_path,
        )
        if path.exists()
    ]
    if args.resume == "never" and existing:
        raise RuntimeError(f"fresh training requested but artifacts already exist: {existing}")
    if args.resume == "require" and not existing:
        raise RuntimeError("resume was required but no resumable artifact exists")
    if (
        diagnostics_path.is_file()
        and checkpoint_path.is_file()
        and provenance_path.is_file()
        and args.resume != "never"
    ):
        provenance = adapter.load_strict_json(provenance_path)
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            raise RuntimeError("completed output has unsupported resumable provenance")
        if provenance.get("resume_binding_sha256") != binding_sha256:
            raise RuntimeError("completed output belongs to different code, inputs, or settings")
        diagnostics = adapter.load_strict_json(diagnostics_path)
        if provenance.get("checkpoint_sha256") != adapter.sha256_file(checkpoint_path):
            raise RuntimeError("completed checkpoint does not match resumable provenance")
        if provenance.get("training_diagnostics_sha256") != adapter.sha256_file(
            diagnostics_path
        ):
            raise RuntimeError("completed diagnostics do not match resumable provenance")
        print(json.dumps(diagnostics, indent=2))
        return

    prior_process_seconds = _prior_process_seconds(progress_path, binding_sha256)
    reporter = ProgressReporter(
        progress_path,
        binding_sha256=binding_sha256,
        source=source,
        runner=runner,
        device=str(device),
        output_dir=output_dir,
        prior_process_seconds=prior_process_seconds,
        invocation_started=invocation_started,
    )

    graph = adapter.line_graph_from_plain(artifact.pop("graph"))
    contraction = adapter.contraction_from_plain(artifact.pop("contraction"))
    adapter.validate_trips(train_dataset.trips, graph)
    adapter.validate_trips(validation_dataset.trips, graph)
    embeddings = adapter.np.asarray(artifact.pop("embeddings"), dtype=adapter.np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != graph.state_count:
        raise RuntimeError("preprocessed embeddings have an invalid shape")
    raw_sc1 = artifact.pop("sc1")
    if not isinstance(raw_sc1, dict):
        raise RuntimeError("preprocessing SC1 artifact is not a dictionary")
    sc1: dict[tuple[int, int], list[int]] = raw_sc1
    artifact_map = artifact.pop("map")
    artifact.clear()

    stage = "loading"
    retained_compute = 0.0
    peak_candidates: list[dict[str, int]] = []
    try:
        _raise_if_stop_requested()
        stage = "original_training"
        original_model = adapter.make_transition_model(
            embeddings.shape[1], args.hidden_dimension
        )

        def original_validation(candidate: Any) -> tuple[list[list[int]], dict[str, Any]]:
            return adapter.greedy_paths(
                candidate,
                embeddings,
                validation_dataset.trips,
                graph.outgoing,
                device,
                inference_batch_size=args.inference_batch_size,
                max_steps=args.max_steps,
            )

        original_model, original_training, original_seconds, original_peaks = (
            fit_transition_model_resumable(
                original_model,
                embeddings,
                graph.outgoing,
                train_dataset.trips,
                original_validation,
                validation_dataset.trips,
                stage=stage,
                stage_index=0,
                resume_path=original_resume,
                binding_sha256=binding_sha256,
                runner=runner,
                reporter=reporter,
                retained_before_seconds=0.0,
                epochs=args.epochs,
                validation_every=args.validation_every,
                batch_size=args.batch_size,
                transition_chunk_size=args.transition_chunk_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=device,
            )
        )
        retained_compute = original_seconds
        peak_candidates.append(original_peaks)

        stage = "sc2"
        sc2, sc2_stats, sc2_seconds, sc2_peaks = build_sc2_database_resumable(
            original_model,
            embeddings,
            graph,
            contraction,
            sc1,
            device,
            score_batch_size=args.sc2_score_batch_size,
            checkpoint_every=args.sc2_checkpoint_every,
            resume_path=sc2_resume,
            binding_sha256=binding_sha256,
            runner=runner,
            reporter=reporter,
            retained_before_seconds=retained_compute,
        )
        retained_compute += sc2_seconds
        peak_candidates.append(sc2_peaks)
        sc1_count = len(sc1)
        overlap = next((pair for pair in sc2 if pair in sc1), None)
        if overlap is not None:
            raise RuntimeError(f"SC1 and SC2 unexpectedly overlap at {overlap}")
        shortcuts = sc1
        shortcuts.update(sc2)
        if len(shortcuts) != len(contraction.shortcut_pairs) or any(
            pair not in shortcuts for pair in contraction.shortcut_pairs
        ):
            raise RuntimeError("SC1/SC2 do not cover every final sparse shortcut")
        for pair, path in shortcuts.items():
            adapter.validate_expansion_path(path, pair, graph)

        sparse_train_trips = adapter.SparseTripView(
            train_dataset.trips, contraction.active
        )
        dropped_sparse_routes = sparse_train_trips.dropped_routes
        if not sparse_train_trips:
            raise RuntimeError("contraction removed every sparse training transition")
        adapter.torch.manual_seed(args.seed + 1)
        sparse_model = adapter.make_transition_model(
            embeddings.shape[1], args.hidden_dimension
        )

        def sparse_validation(candidate: Any) -> tuple[list[list[int]], dict[str, Any]]:
            return adapter.predict_dual_level(
                original_model,
                candidate,
                embeddings,
                graph,
                contraction,
                shortcuts,
                validation_dataset.trips,
                device,
                inference_batch_size=args.inference_batch_size,
                max_steps=args.max_steps,
            )

        stage = "sparse_training"
        sparse_model, sparse_training, sparse_seconds, sparse_peaks = (
            fit_transition_model_resumable(
                sparse_model,
                embeddings,
                contraction.sparse_outgoing,
                sparse_train_trips,
                sparse_validation,
                validation_dataset.trips,
                stage=stage,
                stage_index=2,
                resume_path=sparse_resume,
                binding_sha256=binding_sha256,
                runner=runner,
                reporter=reporter,
                retained_before_seconds=retained_compute,
                epochs=args.epochs,
                validation_every=args.validation_every,
                batch_size=args.batch_size,
                transition_chunk_size=args.transition_chunk_size,
                learning_rate=args.learning_rate,
                seed=args.seed + 1,
                device=device,
            )
        )
        retained_compute += sparse_seconds
        peak_candidates.append(sparse_peaks)
        _raise_if_stop_requested()
        stage = "finalizing"

        final_peaks = _resource_peaks()
        for candidate in peak_candidates:
            final_peaks = _resource_peaks(
                {
                    "peak_rss_kib": max(
                        final_peaks["peak_rss_kib"], candidate["peak_rss_kib"]
                    ),
                    "peak_cuda_memory_bytes": max(
                        final_peaks["peak_cuda_memory_bytes"],
                        candidate["peak_cuda_memory_bytes"],
                    ),
                }
            )
        checkpoint = {
            "schema": adapter.CHECKPOINT_SCHEMA,
            "adapter_version": adapter.ADAPTER_VERSION,
            "audited_upstream_commit": adapter.AUDITED_UPSTREAM_COMMIT,
            "source": source,
            "train_dataset": adapter.dataset_binding(train_dataset),
            "validation_dataset": adapter.dataset_binding(validation_dataset),
            "map": artifact_map,
            "graph": adapter.line_graph_to_plain(graph),
            "contraction": adapter.contraction_to_plain(contraction),
            "embeddings": embeddings,
            "shortcuts": shortcuts,
            "embedding_dimension": int(embeddings.shape[1]),
            "hidden_dimension": args.hidden_dimension,
            "original_model_state": adapter.state_dict_on_cpu(original_model),
            "sparse_model_state": adapter.state_dict_on_cpu(sparse_model),
            "original_selected_epoch": original_training["selected_epoch"],
            "sparse_selected_epoch": sparse_training["selected_epoch"],
            "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
            "configuration": configuration,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter.write_torch_atomic(checkpoint_path, checkpoint)
        cumulative_active_seconds = (
            reporter.prior_process_seconds + time.perf_counter() - invocation_started
        )
        diagnostics = {
            "schema": adapter.TRAINING_DIAGNOSTICS_SCHEMA,
            "method": adapter.METHOD_NAME,
            "adapter_version": adapter.ADAPTER_VERSION,
            "source": source,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": adapter.sha256_file(checkpoint_path),
            "preprocess_dir": str(args.preprocess_dir),
            "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
            "train_manifest": str(train_dataset.manifest_path),
            "train_dataset": adapter.dataset_binding(train_dataset),
            "validation_manifest": str(validation_dataset.manifest_path),
            "validation_dataset": adapter.dataset_binding(validation_dataset),
            "split_roles_read": [
                train_dataset.manifest.split_role,
                validation_dataset.manifest.split_role,
            ],
            "test_data_read": any(
                dataset.manifest.split_role == "test"
                for dataset in (train_dataset, validation_dataset)
            ),
            "graph_identity": graph.identity,
            "configuration": configuration,
            "original_model": original_training,
            "sc2": sc2_stats,
            "sparse_model": sparse_training,
            "shortcut_storage": {
                "final_shortcuts": len(contraction.shortcut_pairs),
                "sc1_train_historical": sc1_count,
                "sc2_model_cost": len(sc2),
            },
            "sparse_training_routes": len(sparse_train_trips),
            "sparse_training_routes_dropped_below_two_states": dropped_sparse_routes,
            "sparse_training_index_storage_bytes": sparse_train_trips.index_storage_bytes,
            "total_process_seconds": cumulative_active_seconds,
            "peak_rss_kib": final_peaks["peak_rss_kib"],
            "peak_cuda_memory_bytes": final_peaks["peak_cuda_memory_bytes"],
            "environment": adapter.environment_info(str(device), workers),
        }
        adapter.write_json_atomic(diagnostics_path, diagnostics)
        final_invocation_seconds = time.perf_counter() - invocation_started
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "runner": runner,
            "adapter_source": source,
            "resume_binding_sha256": binding_sha256,
            "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
            "train_dataset": adapter.dataset_binding(train_dataset),
            "validation_dataset": adapter.dataset_binding(validation_dataset),
            "configuration": configuration,
            "progress": str(progress_path),
            "resume_dir": str(resume_dir),
            "stage_checkpoints": {
                "original": {
                    "path": str(original_resume),
                    "sha256": adapter.sha256_file(original_resume),
                },
                "sc2": {
                    "path": str(sc2_resume),
                    "sha256": adapter.sha256_file(sc2_resume),
                },
                "sparse": {
                    "path": str(sparse_resume),
                    "sha256": adapter.sha256_file(sparse_resume),
                },
            },
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": adapter.sha256_file(checkpoint_path),
            "training_diagnostics": str(diagnostics_path),
            "training_diagnostics_sha256": adapter.sha256_file(diagnostics_path),
            "timing": {
                "prior_invocations_active_seconds": reporter.prior_process_seconds,
                "final_invocation_active_seconds": final_invocation_seconds,
                "cumulative_active_process_seconds_at_diagnostics": (
                    cumulative_active_seconds
                ),
                "cumulative_active_process_seconds_at_provenance": (
                    reporter.prior_process_seconds + final_invocation_seconds
                ),
                "retained_stage_compute_seconds": retained_compute,
                "active_overhead_and_unretained_work_seconds": max(
                    0.0, cumulative_active_seconds - retained_compute
                ),
            },
            "completed_at": utc_now(),
        }
        adapter.write_json_atomic(provenance_path, provenance)
        reporter.write(
            status="complete",
            stage="complete",
            stage_index=3,
            completed_units=1,
            total_units=1,
            unit="run",
            phase_elapsed_seconds=0.0,
            retained_compute_seconds=retained_compute,
            checkpoint=checkpoint_path,
        )
        print(json.dumps(diagnostics, indent=2))
    except StopRequested:
        raise
    except Exception as error:
        last = reporter.last or {}
        reporter.write(
            status="failed",
            stage=stage,
            stage_index=int(last.get("stage_index", 0)),
            completed_units=int(last.get("completed_units", 0)),
            recoverable_completed_units=int(
                last.get("recoverable_completed_units", 0)
            ),
            total_units=int(last.get("total_units", 0)),
            unit=str(last.get("unit", "unknown")),
            phase_elapsed_seconds=float(last.get("phase_elapsed_seconds", 0.0)),
            retained_compute_seconds=retained_compute,
            message=f"{type(error).__name__}: {error}",
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable GPU-capable trainer for the DRNCS-LG adapter"
    )
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--expected-train-manifest-sha256")
    parser.add_argument("--expected-train-records-sha256")
    parser.add_argument("--expected-validation-manifest-sha256")
    parser.add_argument("--expected-validation-records-sha256")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--transition-chunk-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--sc2-score-batch-size", type=int, default=4096)
    parser.add_argument("--resume", choices=["auto", "never", "require"], default="auto")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--sc2-checkpoint-every", type=int, default=25)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    previous_int = signal.signal(signal.SIGINT, _signal_handler)
    previous_term = signal.signal(signal.SIGTERM, _signal_handler)
    try:
        train_command(parse_args(argv))
    except StopRequested as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(128 + int(_STOP_SIGNAL or signal.SIGINT)) from error
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
