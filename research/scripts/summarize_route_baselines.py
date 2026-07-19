#!/usr/bin/env python3
"""Build JSON, CSV, and Markdown route-baseline paper tables from artifacts.

The summarizer is intentionally conservative: a value is emitted only when a
versioned input artifact contains enough information to support it.  Missing
runs stay ``pending`` and missing cells stay JSON ``null`` / Markdown ``—``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


INPUT_SCHEMA = "ewr.route-baseline-summary-input/v1"
OUTPUT_SCHEMA = "ewr.route-baseline-summary/v1"
EVALUATION_SCHEMA = "ewr.evaluation-summary/v1"
SHARDED_QUALITY_PREDICTION_SCHEMA = (
    "ewr.sharded-quality-prediction-diagnostics/v1"
)
NEUROMLR_CHUNKED_QUALITY_MODE = "chunked_resumable_quality_prediction"
OPERATIONAL_EFFICIENCY_SCHEMA = "ewr.full-test-operational-efficiency/v1"

METHODS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("project", "Project"),
        ("sp_length", "SP-Length"),
        ("markov_sp", "Markov-SP"),
        ("neuromlr_greedy", "NeuroMLR-G"),
        ("drncs_lg", "DRNCS-LG"),
        ("drpk_static", "DRPK-static"),
        ("drp_tp", "DRP-TP"),
    )
)

METHOD_INPUT_KEYS = {
    "evaluation",
    "offline",
    "training",
    "prediction",
    "operational_prediction",
    "artifacts",
}
METRIC_KEYS = {
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "edge_jaccard",
    "exact_match",
}

ARTIFACT_KINDS = {"model", "auxiliary"}
MODEL_ARTIFACT_METHODS = {
    "project", "sp_length", "markov_sp", "neuromlr_greedy",
    "drncs_lg", "drpk_static",
}
AUXILIARY_ARTIFACT_METHODS = {"drpk_static", "drp_tp"}

# The Project predictor predates the study registry and wrote its implementation
# name (``project_cch``) into the v1 diagnostics.  Treat that one historical ID
# as an explicit schema-scoped alias; all other artifact methods still have to
# match their registry ID exactly.
PROJECT_PREDICTION_METHOD_IDS = frozenset({"project", "project_cch"})

# Completeness is method-aware.  A method is not penalized for a phase that is
# not part of its algorithm (for example, DRP-TP has no fitted training phase),
# but a phase that is part of the registered protocol must carry both time and
# peak RSS evidence.  NeuroMLR's historical adapter does not expose a CPU worker
# count, so threads are not treated as an applicable inference field for it.
EFFICIENCY_PHASES: dict[str, tuple[str, ...]] = {
    "project": ("training", "prediction"),
    "sp_length": ("offline", "prediction"),
    "markov_sp": ("training", "prediction"),
    "neuromlr_greedy": ("training", "prediction"),
    "drncs_lg": ("offline", "training", "prediction"),
    "drpk_static": ("offline", "training", "prediction"),
    "drp_tp": ("offline", "prediction"),
}
INFERENCE_THREADS_APPLICABLE = set(METHODS) - {"neuromlr_greedy"}

# Only stable, study-owned filenames are auto-discovered.  In particular, the
# script never recursively guesses which Project/DRNCS/DRPK file is formal.
AUTO_ARTIFACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "sp_length": {
        "evaluation": (
            "static/sp_length.evaluation.json",
            "static/sp-length.test.evaluation.json",
        ),
        "training": (
            "static/sp_length.train.diagnostics.json",
            "static/sp-length.train.json",
        ),
        "prediction": (
            "static/sp_length.test.diagnostics.json",
            "static/sp-length.predict.json",
        ),
    },
    "markov_sp": {
        "evaluation": (
            "static/markov_sp.evaluation.json",
            "static/markov-sp.test.evaluation.json",
        ),
        "training": (
            "static/markov_sp.train.diagnostics.json",
            "static/markov-sp.train.json",
        ),
        "prediction": (
            "static/markov_sp.test.diagnostics.json",
            "static/markov-sp.predict.json",
        ),
    },
    "neuromlr_greedy": {
        "evaluation": (
            "neuromlr/greedy.test.evaluation.json",
            "neuromlr/greedy.evaluation.json",
        ),
        "prediction": ("neuromlr/greedy.test.diagnostics.json",),
    },
}

DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "archive/experiments/neuromlr_cch_dijkstra_benchmarks/summary.json"
)


class SchemaError(ValueError):
    """An input JSON document does not conform to its declared schema."""


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SchemaError(f"{path}: top-level JSON value must be an object")
    return value


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{context} must be an object")
    return value


def _exact(
    value: Any,
    required: Iterable[str],
    context: str,
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    obj = _object(value, context)
    required_keys, optional_keys = set(required), set(optional)
    missing = sorted(required_keys - obj.keys())
    extra = sorted(obj.keys() - required_keys - optional_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise SchemaError(f"{context}: {'; '.join(details)}")
    return obj


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be a nonempty string")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{context} must be a boolean")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise SchemaError(f"{context} must be finite and >= {minimum}")
    return converted


def _nullable_integer(value: Any, context: str) -> int | None:
    return None if value is None else _integer(value, context)


def _nullable_number(value: Any, context: str) -> float | None:
    return None if value is None else _number(value, context)


def _number_list(value: Any, context: str) -> list[float]:
    if not isinstance(value, list):
        raise SchemaError(f"{context} must be an array")
    return [_number(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _schema(value: Mapping[str, Any], expected: str, context: str) -> None:
    observed = _string(value.get("schema"), f"{context}.schema")
    if observed != expected:
        raise SchemaError(
            f"{context}.schema is {observed!r}; expected {expected!r}"
        )


def validate_evaluation(value: Any, context: str) -> dict[str, Any]:
    obj = _exact(value, {"schema", "sample_count", "metrics"}, context)
    _schema(obj, EVALUATION_SCHEMA, context)
    count = _integer(obj["sample_count"], f"{context}.sample_count", minimum=1)
    metrics = _exact(obj["metrics"], METRIC_KEYS, f"{context}.metrics")
    result: dict[str, Any] = {"sample_count": count}
    for key in sorted(METRIC_KEYS):
        metric = _number(metrics[key], f"{context}.metrics.{key}")
        if metric > 1.0:
            raise SchemaError(f"{context}.metrics.{key} must be <= 1")
        result[key] = metric
    return result


def _validate_repetitions(
    timing: Mapping[str, Any],
    warmups: int,
    measured: int,
    warmup_key: str,
    measured_key: str,
    context: str,
) -> None:
    warmup_values = _number_list(timing[warmup_key], f"{context}.{warmup_key}")
    measured_values = _number_list(timing[measured_key], f"{context}.{measured_key}")
    if len(warmup_values) != warmups or len(measured_values) != measured:
        raise SchemaError(f"{context}: repetition arrays do not match declared counts")


def validate_static_training(
    obj: dict[str, Any], expected_method: str, context: str
) -> dict[str, Any]:
    keys = {
        "schema", "method", "query_protocol", "threads", "training_samples",
        "validation_samples", "coordinate_count", "transition_observations",
        "observed_coordinates", "selected_alpha", "validation_candidates",
        "timing", "peak_rss_kib",
    }
    obj = _exact(obj, keys, context)
    _schema(obj, "ewr.static-route-baseline-training-diagnostics/v1", context)
    if obj["method"] != expected_method:
        raise SchemaError(f"{context}.method does not match {expected_method!r}")
    _string(obj["query_protocol"], f"{context}.query_protocol")
    _integer(obj["threads"], f"{context}.threads", minimum=1)
    for key in (
        "training_samples", "validation_samples", "coordinate_count",
        "transition_observations", "observed_coordinates", "peak_rss_kib",
    ):
        _integer(obj[key], f"{context}.{key}")
    timing_keys = {
        "network_and_topology_seconds", "training_records_load_seconds",
        "transition_counting_seconds", "validation_selection_seconds",
        "total_before_artifact_write_seconds",
    }
    timing = _exact(obj["timing"], timing_keys, f"{context}.timing")
    for key in timing_keys:
        _number(timing[key], f"{context}.timing.{key}")
    _nullable_number(obj["selected_alpha"], f"{context}.selected_alpha")
    if not isinstance(obj["validation_candidates"], list):
        raise SchemaError(f"{context}.validation_candidates must be an array")
    candidate_keys = {
        "alpha", "quantization_scale", "routing_seconds", "exact_match",
        "edge_precision", "edge_recall", "edge_f1", "edge_jaccard",
    }
    for index, raw_candidate in enumerate(obj["validation_candidates"]):
        candidate_context = f"{context}.validation_candidates[{index}]"
        candidate = _exact(raw_candidate, candidate_keys, candidate_context)
        _nullable_number(candidate["alpha"], f"{candidate_context}.alpha")
        for key in ("quantization_scale", "routing_seconds"):
            _number(candidate[key], f"{candidate_context}.{key}")
        for key in METRIC_KEYS:
            metric = _number(candidate[key], f"{candidate_context}.{key}")
            if metric > 1.0:
                raise SchemaError(f"{candidate_context}.{key} must be <= 1")
    return obj


def validate_project_training(obj: dict[str, Any], context: str) -> dict[str, Any]:
    keys = {
        "schema", "accepted", "dropped", "coordinates", "completed_updates",
        "objective", "threads", "checkpoint_every", "snapshots", "timing",
        "peak_rss_kib",
    }
    obj = _exact(obj, keys, context)
    _schema(obj, "ewr.project-training-summary/v1", context)
    for key in ("accepted", "dropped", "coordinates", "completed_updates"):
        _integer(obj[key], f"{context}.{key}")
    _integer(obj["threads"], f"{context}.threads", minimum=1)
    _number(obj["objective"], f"{context}.objective")
    timing_keys = {
        "input_load_seconds", "setup_training_and_snapshot_seconds",
        "total_before_summary_write_seconds",
    }
    timing = _exact(obj["timing"], timing_keys, f"{context}.timing")
    for key in timing_keys:
        _number(timing[key], f"{context}.timing.{key}")
    _nullable_integer(obj["peak_rss_kib"], f"{context}.peak_rss_kib")
    if not isinstance(obj["snapshots"], list):
        raise SchemaError(f"{context}.snapshots must be an array")
    return obj


def validate_neuromlr_selection(obj: dict[str, Any], context: str) -> dict[str, Any]:
    keys = {
        "schema_version", "evaluations", "peak_rss_kib", "selected",
        "selection_rule", "test_read", "total_seconds", "training_seconds",
    }
    obj = _exact(obj, keys, context)
    if _integer(obj["schema_version"], f"{context}.schema_version") != 1:
        raise SchemaError(f"{context}.schema_version must equal 1")
    _boolean(obj["test_read"], f"{context}.test_read")
    for key in ("peak_rss_kib",):
        _integer(obj[key], f"{context}.{key}")
    for key in ("total_seconds", "training_seconds"):
        _number(obj[key], f"{context}.{key}")
    if not isinstance(obj["evaluations"], list):
        raise SchemaError(f"{context}.evaluations must be an array")
    _object(obj["selected"], f"{context}.selected")
    return obj


def validate_training(
    value: dict[str, Any], method: str, context: str
) -> dict[str, Any]:
    schema = value.get("schema")
    if schema == "ewr.static-route-baseline-training-diagnostics/v1":
        if method not in {"sp_length", "markov_sp"}:
            raise SchemaError(f"{context}: static training is invalid for {method}")
        return validate_static_training(value, method, context)
    if schema == "ewr.project-training-summary/v1":
        if method != "project":
            raise SchemaError(f"{context}: project training is invalid for {method}")
        return validate_project_training(value, context)
    if schema in {
        "ewr.drncs-lg-training-diagnostics/v1",
        "ewr.drncs-lg-training-diagnostics/v2",
    }:
        if method != "drncs_lg":
            raise SchemaError(f"{context}: DRNCS training is invalid for {method}")
        keys = {
            "schema", "method", "adapter_version", "checkpoint",
            "checkpoint_sha256", "preprocess_dir", "preprocess_artifact_sha256",
            "train_manifest", "train_dataset", "validation_manifest",
            "validation_dataset", "test_data_read", "graph_identity",
            "configuration", "original_model", "sc2", "sparse_model",
            "shortcut_storage", "sparse_training_routes",
            "sparse_training_routes_dropped_below_two_states",
            "total_process_seconds", "peak_rss_kib", "peak_cuda_memory_bytes",
            "environment",
        }
        if schema.endswith("/v2"):
            obj = _exact(
                value,
                keys
                | {
                    "source",
                    "split_roles_read",
                    "sparse_training_index_storage_bytes",
                },
                context,
            )
            source = _object(obj["source"], f"{context}.source")
            _sha256_string(
                source.get("adapter_sha256"),
                f"{context}.source.adapter_sha256",
            )
            if obj["split_roles_read"] != ["train", "validation"]:
                raise SchemaError(
                    f"{context}.split_roles_read must be ['train', 'validation']"
                )
            _integer(
                obj["sparse_training_index_storage_bytes"],
                f"{context}.sparse_training_index_storage_bytes",
            )
        else:
            obj = _exact(value, keys, context)
        if obj["method"] != method:
            raise SchemaError(f"{context}.method does not match {method!r}")
        _schema(obj, schema, context)
        _boolean(obj["test_data_read"], f"{context}.test_data_read")
        _number(obj["total_process_seconds"], f"{context}.total_process_seconds")
        for key in ("peak_rss_kib", "peak_cuda_memory_bytes"):
            _integer(obj[key], f"{context}.{key}")
        configuration = _object(obj["configuration"], f"{context}.configuration")
        _integer(configuration.get("workers"), f"{context}.configuration.workers", minimum=1)
        _string(configuration.get("device"), f"{context}.configuration.device")
        environment = _object(obj["environment"], f"{context}.environment")
        _string(environment.get("device"), f"{context}.environment.device")
        return obj
    if schema in {"ewr.drpk-static-selection/v1", "ewr.drpk-static-selection/v2"}:
        if method != "drpk_static":
            raise SchemaError(f"{context}: DRPK training is invalid for {method}")
        v1_keys = {
            "schema", "selection_rule", "selected", "best_loss", "evaluations",
            "epochs_completed", "training_seconds", "total_seconds", "peak_rss_kib",
        }
        if schema.endswith("/v1"):
            obj = _exact(value, v1_keys, context)
        else:
            obj = _exact(
                value,
                v1_keys | {
                    "resolved_device", "workers", "environment", "checkpoint_last",
                    "peak_cuda_memory_bytes",
                },
                context,
                {
                    "optimizer_steps",
                    "microbatches",
                    "requested_device",
                    "source",
                },
            )
            _string(obj["resolved_device"], f"{context}.resolved_device")
            if "requested_device" in obj:
                _string(obj["requested_device"], f"{context}.requested_device")
            if "source" in obj:
                source = _object(obj["source"], f"{context}.source")
                _sha256_string(
                    source.get("adapter_sha256"),
                    f"{context}.source.adapter_sha256",
                )
            _integer(obj["workers"], f"{context}.workers", minimum=1)
            environment = _object(obj["environment"], f"{context}.environment")
            environment_device = _string(
                environment.get("device"), f"{context}.environment.device"
            )
            if environment_device != obj["resolved_device"]:
                raise SchemaError(
                    f"{context}: resolved_device and environment.device differ"
                )
            checkpoint_last = _exact(
                obj["checkpoint_last"], {"path", "sha256", "epoch"},
                f"{context}.checkpoint_last",
            )
            _string(checkpoint_last["path"], f"{context}.checkpoint_last.path")
            _sha256_string(
                checkpoint_last["sha256"], f"{context}.checkpoint_last.sha256"
            )
            _integer(checkpoint_last["epoch"], f"{context}.checkpoint_last.epoch")
            _integer(
                obj["peak_cuda_memory_bytes"],
                f"{context}.peak_cuda_memory_bytes",
            )
            for optional_count in ("optimizer_steps", "microbatches"):
                if optional_count in obj:
                    _integer(
                        obj[optional_count], f"{context}.{optional_count}", minimum=1
                    )
            for selection_name in ("selected", "best_loss"):
                selected = _object(
                    obj[selection_name], f"{context}.{selection_name}"
                )
                _string(
                    selected.get("checkpoint"),
                    f"{context}.{selection_name}.checkpoint",
                )
                _sha256_string(
                    selected.get("checkpoint_sha256"),
                    f"{context}.{selection_name}.checkpoint_sha256",
                )
        _schema(obj, schema, context)
        for key in ("training_seconds", "total_seconds"):
            _number(obj[key], f"{context}.{key}")
        for key in ("epochs_completed", "peak_rss_kib"):
            _integer(obj[key], f"{context}.{key}")
        if not isinstance(obj["evaluations"], list):
            raise SchemaError(f"{context}.evaluations must be an array")
        if not isinstance(obj["selection_rule"], list):
            raise SchemaError(f"{context}.selection_rule must be an array")
        return obj
    if "schema_version" in value and method == "neuromlr_greedy":
        return validate_neuromlr_selection(value, context)
    raise SchemaError(f"{context}: unsupported training schema {schema!r}")


def validate_offline(value: dict[str, Any], method: str, context: str) -> dict[str, Any]:
    schema = value.get("schema")
    if schema in {"ewr.drncs-lg-preprocess/v1", "ewr.drncs-lg-preprocess/v2"}:
        if method != "drncs_lg":
            raise SchemaError(f"{context}: DRNCS preprocessing is invalid for {method}")
        keys = {
            "schema", "adapter_version", "artifact", "artifact_sha256",
            "train_manifest", "train_dataset", "map", "graph_identity", "states",
            "transitions", "contracted_states", "surviving_states", "final_shortcuts",
            "sc1", "configuration", "timing", "peak_rss_kib", "environment",
        }
        obj = _exact(
            value,
            keys | ({"source"} if schema.endswith("/v2") else set()),
            context,
        )
        _schema(obj, schema, context)
        if schema.endswith("/v2"):
            source = _object(obj["source"], f"{context}.source")
            _sha256_string(
                source.get("adapter_sha256"),
                f"{context}.source.adapter_sha256",
            )
        timing_keys = {
            "data_and_graph_seconds", "node2vec_seconds", "contraction_seconds",
            "sc1_seconds", "total_process_seconds",
        }
        if schema.endswith("/v2"):
            timing_keys |= {
                "map_and_graph_seconds",
                "train_data_load_and_validation_seconds",
            }
        timing = _exact(obj["timing"], timing_keys, f"{context}.timing")
        for key in timing_keys:
            _number(timing[key], f"{context}.timing.{key}")
        _integer(obj["peak_rss_kib"], f"{context}.peak_rss_kib")
        configuration = _object(obj["configuration"], f"{context}.configuration")
        _integer(configuration.get("workers"), f"{context}.configuration.workers", minimum=1)
        environment = _object(obj["environment"], f"{context}.environment")
        _string(environment.get("device"), f"{context}.environment.device")
        return obj
    if schema in {
        "ewr.drpk-static-preprocess-diagnostics/v1",
        "ewr.drpk-static-preprocess-diagnostics/v2",
    }:
        if method not in {"drpk_static", "drp_tp"}:
            raise SchemaError(f"{context}: DRPK preprocessing is invalid for {method}")
        obj = _exact(
            value, {"schema", "configuration", "counts", "timing", "peak_rss_kib"}, context
        )
        _schema(obj, schema, context)
        legacy_timing_keys = {
            "graph_seconds", "da_and_popularity_seconds",
            "candidate_label_seconds", "node2vec", "total_seconds",
        }
        if schema.endswith("/v2"):
            timing = _exact(
                obj["timing"],
                {
                    "graph_seconds", "da_seconds", "popularity_seconds",
                    "candidate_label_seconds", "node2vec", "total_seconds",
                    "drp_tp_ready_seconds", "drp_tp_ready_peak_rss_kib",
                },
                f"{context}.timing",
            )
        else:
            timing = _exact(
                obj["timing"], legacy_timing_keys, f"{context}.timing",
                {"drp_tp_ready_seconds", "drp_tp_ready_peak_rss_kib"},
            )
        _number(timing["total_seconds"], f"{context}.timing.total_seconds")
        ready_fields = {
            "drp_tp_ready_seconds", "drp_tp_ready_peak_rss_kib"
        } & timing.keys()
        if ready_fields and len(ready_fields) != 2:
            raise SchemaError(
                f"{context}.timing must declare both DRP-TP readiness fields"
            )
        if ready_fields:
            _number(
                timing["drp_tp_ready_seconds"],
                f"{context}.timing.drp_tp_ready_seconds",
            )
            _integer(
                timing["drp_tp_ready_peak_rss_kib"],
                f"{context}.timing.drp_tp_ready_peak_rss_kib",
            )
        _integer(obj["peak_rss_kib"], f"{context}.peak_rss_kib")
        configuration = _object(obj["configuration"], f"{context}.configuration")
        _integer(configuration.get("workers"), f"{context}.configuration.workers", minimum=1)
        return obj
    raise SchemaError(f"{context}: unsupported offline schema {schema!r}")


def _validate_prediction_timing(
    obj: dict[str, Any], timing_keys: set[str], context: str
) -> dict[str, Any]:
    timing = _exact(obj["timing"], timing_keys, f"{context}.timing")
    for key in ("mean_seconds_per_query", "queries_per_second"):
        _number(timing[key], f"{context}.timing.{key}")
    return timing


def _validate_sharded_quality_prediction(
    value: dict[str, Any], method: str, context: str
) -> dict[str, Any]:
    supported = {"drncs_lg", "drpk_static", "drp_tp"}
    obj = _exact(
        value,
        {
            "schema", "method", "purpose", "efficiency_comparable",
            "efficiency_exclusion_reason", "binding_sha256", "dataset",
            "configuration", "shards", "samples", "endpoint_failures",
            "generated_route_validity", "operational_timing",
            "maximum_shard_peak_rss_kib",
            "maximum_shard_peak_cuda_memory_bytes", "predictions",
            "shard_outputs", "completed_at",
        },
        context,
    )
    _schema(obj, SHARDED_QUALITY_PREDICTION_SCHEMA, context)
    if method not in supported or obj["method"] != method:
        raise SchemaError(f"{context}.method does not match {method!r}")
    if obj["purpose"] != "full_test_quality_prediction_only":
        raise SchemaError(f"{context}.purpose is not full-test quality prediction")
    if _boolean(
        obj["efficiency_comparable"], f"{context}.efficiency_comparable"
    ):
        raise SchemaError(f"{context}: sharded quality timing is not comparable")
    _string(
        obj["efficiency_exclusion_reason"],
        f"{context}.efficiency_exclusion_reason",
    )
    _sha256_string(obj["binding_sha256"], f"{context}.binding_sha256")

    dataset = _exact(
        obj["dataset"],
        {
            "dataset_id", "network_id", "manifest_path", "manifest_sha256",
            "records_path", "records_sha256", "samples", "first_sample_id",
            "last_sample_id",
        },
        f"{context}.dataset",
    )
    for key in (
        "dataset_id", "network_id", "manifest_path", "records_path",
        "first_sample_id", "last_sample_id",
    ):
        _string(dataset[key], f"{context}.dataset.{key}")
    for key in ("manifest_sha256", "records_sha256"):
        _sha256_string(dataset[key], f"{context}.dataset.{key}")
    samples = _integer(obj["samples"], f"{context}.samples", minimum=1)
    if _integer(
        dataset["samples"], f"{context}.dataset.samples", minimum=1
    ) != samples:
        raise SchemaError(f"{context}: dataset samples differ from samples")
    failures = _integer(
        obj["endpoint_failures"], f"{context}.endpoint_failures"
    )
    if failures > samples:
        raise SchemaError(f"{context}.endpoint_failures exceeds samples")

    configuration = _exact(
        obj["configuration"],
        {
            "shard_size", "seed", "workers", "device",
            "cuda_visible_devices", "inference_batch_size", "max_steps",
            "warmup_repetitions", "measured_repetitions", "latency_samples",
            "purpose", "efficiency_table_source",
        },
        f"{context}.configuration",
    )
    _integer(configuration["shard_size"], f"{context}.configuration.shard_size", minimum=1)
    _integer(configuration["seed"], f"{context}.configuration.seed")
    _integer(configuration["workers"], f"{context}.configuration.workers", minimum=1)
    _integer(
        configuration["inference_batch_size"],
        f"{context}.configuration.inference_batch_size",
        minimum=1,
    )
    if configuration["warmup_repetitions"] != 0 or configuration["measured_repetitions"] != 1:
        raise SchemaError(
            f"{context}: sharded quality prediction must use 0 warm-ups and 1 repetition"
        )
    if configuration["purpose"] != obj["purpose"]:
        raise SchemaError(f"{context}: configuration purpose differs")
    if configuration["efficiency_table_source"] != "current_full_test_single_pass_outer_wall":
        raise SchemaError(f"{context}: wrong efficiency table source")
    device = _string(configuration["device"], f"{context}.configuration.device")
    if method == "drp_tp":
        if device != "cpu" or configuration["cuda_visible_devices"] is not None:
            raise SchemaError(f"{context}: DRP-TP sharded prediction must use CPU")
    else:
        if not device.startswith("cuda"):
            raise SchemaError(f"{context}: {method} sharded prediction must use CUDA")
        cuda_visible = configuration["cuda_visible_devices"]
        if str(cuda_visible) != "0":
            raise SchemaError(f"{context}: CUDA device visibility must be pinned to 0")
    if method == "drncs_lg":
        _integer(configuration["max_steps"], f"{context}.configuration.max_steps", minimum=1)
        if configuration["latency_samples"] != 0:
            raise SchemaError(f"{context}: quality prediction latency_samples must be 0")
    elif configuration["max_steps"] is not None or configuration["latency_samples"] is not None:
        raise SchemaError(f"{context}: inapplicable method settings must be null")

    validity = obj["generated_route_validity"]
    if validity is not None:
        validity = _object(validity, f"{context}.generated_route_validity")
        for key, value in validity.items():
            _string(key, f"{context}.generated_route_validity key")
            _integer(value, f"{context}.generated_route_validity.{key}")

    operational = _exact(
        obj["operational_timing"],
        {"sum_adapter_process_seconds", "sum_adapter_prediction_seconds"},
        f"{context}.operational_timing",
    )
    for key in operational:
        value = _number(operational[key], f"{context}.operational_timing.{key}")
        if value < 0:
            raise SchemaError(f"{context}.operational_timing.{key} must be nonnegative")
    for key in (
        "maximum_shard_peak_rss_kib", "maximum_shard_peak_cuda_memory_bytes"
    ):
        _integer(obj[key], f"{context}.{key}")
    if method != "drp_tp" and obj["maximum_shard_peak_cuda_memory_bytes"] <= 0:
        raise SchemaError(f"{context}: CUDA prediction reported no CUDA allocation")

    predictions = _exact(
        obj["predictions"], {"path", "sha256", "records_schema"},
        f"{context}.predictions",
    )
    _string(predictions["path"], f"{context}.predictions.path")
    _sha256_string(predictions["sha256"], f"{context}.predictions.sha256")
    if predictions["records_schema"] != "ewr.prediction-record/v1":
        raise SchemaError(f"{context}: wrong prediction records schema")

    shards = _integer(obj["shards"], f"{context}.shards", minimum=1)
    shard_outputs = obj["shard_outputs"]
    if not isinstance(shard_outputs, list) or len(shard_outputs) != shards:
        raise SchemaError(f"{context}.shard_outputs length differs from shards")
    shard_samples = 0
    for index, raw_shard in enumerate(shard_outputs):
        shard = _exact(
            raw_shard,
            {
                "index", "samples", "predictions_sha256", "diagnostics_sha256",
                "run_receipt_sha256", "adapter_environment_sha256",
                "adapter_wall_seconds",
            },
            f"{context}.shard_outputs[{index}]",
        )
        if shard["index"] != index:
            raise SchemaError(f"{context}.shard_outputs[{index}].index differs")
        shard_samples += _integer(
            shard["samples"], f"{context}.shard_outputs[{index}].samples", minimum=1
        )
        for key in (
            "predictions_sha256", "diagnostics_sha256", "run_receipt_sha256",
            "adapter_environment_sha256",
        ):
            _sha256_string(shard[key], f"{context}.shard_outputs[{index}].{key}")
        wall = _number(
            shard["adapter_wall_seconds"],
            f"{context}.shard_outputs[{index}].adapter_wall_seconds",
        )
        if wall < 0:
            raise SchemaError(f"{context}: adapter wall time must be nonnegative")
    if shard_samples != samples:
        raise SchemaError(f"{context}: shard samples do not sum to samples")
    _string(obj["completed_at"], f"{context}.completed_at")

    # These compatibility keys let the common quality path consume the record
    # count and endpoint failures without treating shard-internal timings as
    # the formal full-task operational efficiency measurement.
    obj["warmup_repetitions"] = 0
    obj["measured_repetitions"] = 1
    obj["peak_rss_kib"] = obj["maximum_shard_peak_rss_kib"]
    obj["peak_cuda_memory_bytes"] = obj["maximum_shard_peak_cuda_memory_bytes"]
    return obj


def validate_prediction(
    value: dict[str, Any], method: str, context: str
) -> dict[str, Any]:
    schema = value.get("schema")
    if schema == SHARDED_QUALITY_PREDICTION_SCHEMA:
        return _validate_sharded_quality_prediction(value, method, context)
    if schema == "ewr.static-route-baseline-prediction-diagnostics/v1":
        keys = {
            "schema", "method", "query_protocol", "samples", "threads",
            "warmup_repetitions", "measured_repetitions", "selected_alpha",
            "topology_id", "deterministic_repetitions", "endpoint_mismatches",
            "timing", "peak_rss_kib",
        }
        timing_keys = {
            "input_and_network_load_seconds", "topology_and_query_preparation_seconds",
            "warmup_metric_and_query_seconds", "measured_metric_and_query_seconds",
            "mean_metric_and_query_seconds", "mean_seconds_per_query",
            "queries_per_second", "total_before_diagnostics_write_seconds",
            "timing_boundary",
        }
        obj = _exact(value, keys, context)
        _schema(obj, "ewr.static-route-baseline-prediction-diagnostics/v1", context)
        if method not in {"sp_length", "markov_sp"} or obj["method"] != method:
            raise SchemaError(f"{context}.method does not match {method!r}")
        _integer(obj["threads"], f"{context}.threads", minimum=1)
        endpoint_mismatches = _integer(
            obj["endpoint_mismatches"], f"{context}.endpoint_mismatches"
        )
        if endpoint_mismatches > _integer(obj["samples"], f"{context}.samples", minimum=1):
            raise SchemaError(f"{context}.endpoint_mismatches exceeds samples")
        _boolean(obj["deterministic_repetitions"], f"{context}.deterministic_repetitions")
        timing = _validate_prediction_timing(obj, timing_keys, context)
        warmups = _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
        measured = _integer(obj["measured_repetitions"], f"{context}.measured_repetitions", minimum=1)
        _validate_repetitions(timing, warmups, measured, "warmup_metric_and_query_seconds", "measured_metric_and_query_seconds", f"{context}.timing")
    elif schema == "ewr.project-prediction-diagnostics/v1":
        keys = {
            "schema", "method", "query_protocol", "samples", "threads",
            "warmup_repetitions", "measured_repetitions", "completed_updates",
            "objective", "topology_id", "oracle_identity", "deterministic_repetitions",
            "timing", "peak_rss_kib",
        }
        timing_keys = {
            "input_and_network_adapter_load_seconds", "line_graph_and_query_preparation_seconds",
            "warmup_metric_and_query_seconds", "measured_metric_and_query_seconds",
            "mean_metric_and_query_seconds", "mean_seconds_per_query",
            "queries_per_second", "total_before_diagnostics_write_seconds",
            "timing_boundary",
        }
        obj = _exact(value, keys, context)
        _schema(obj, "ewr.project-prediction-diagnostics/v1", context)
        if method != "project" or obj["method"] not in PROJECT_PREDICTION_METHOD_IDS:
            raise SchemaError(f"{context}.method does not match {method!r}")
        _integer(obj["threads"], f"{context}.threads", minimum=1)
        _boolean(obj["deterministic_repetitions"], f"{context}.deterministic_repetitions")
        timing = _validate_prediction_timing(obj, timing_keys, context)
        warmups = _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
        measured = _integer(obj["measured_repetitions"], f"{context}.measured_repetitions", minimum=1)
        _validate_repetitions(timing, warmups, measured, "warmup_metric_and_query_seconds", "measured_metric_and_query_seconds", f"{context}.timing")
    elif schema == "ewr.neuromlr-diagnostics/v1":
        legacy_keys = {
            "schema", "method", "upstream_commit", "checkpoint", "checkpoint_epoch",
            "dataset_manifest", "dataset_manifest_sha256", "dataset_id", "network_id",
            "samples", "query_protocol", "timing", "peak_rss_kib",
            "peak_cuda_memory_bytes", "warmup_repetitions", "measured_repetitions",
            "seed", "traffic",
        }
        extended_keys = legacy_keys | {
            "checkpoint_sha256", "dataset_records", "dataset_records_sha256",
            "graph_identity", "coordinate_identity", "endpoint_failures",
            "predictions_sha256", "execution",
        }
        timing_keys = {
            "data_and_graph_seconds", "model_load_seconds", "prediction_seconds",
            "warmup_repetition_seconds", "prediction_repetition_seconds",
            "mean_seconds_per_query", "queries_per_second", "component_totals",
            "component_totals_per_repetition", "total_process_seconds",
        }
        extended = "execution" in value
        obj = _exact(
            value,
            extended_keys if extended else legacy_keys,
            context,
            () if extended else {"endpoint_failures"},
        )
        _schema(obj, "ewr.neuromlr-diagnostics/v1", context)
        if method != "neuromlr_greedy" or obj["method"] != method:
            raise SchemaError(f"{context}.method does not match {method!r}")
        if extended:
            for key in (
                "checkpoint_sha256", "dataset_manifest_sha256",
                "dataset_records_sha256", "predictions_sha256",
            ):
                _sha256_string(obj[key], f"{context}.{key}")
            for key in (
                "checkpoint", "dataset_manifest", "dataset_records", "dataset_id",
                "network_id", "graph_identity", "coordinate_identity",
            ):
                _string(obj[key], f"{context}.{key}")
            execution = _object(obj["execution"], f"{context}.execution")
            mode = execution.get("mode")
            if mode == NEUROMLR_CHUNKED_QUALITY_MODE:
                execution = _exact(
                    execution,
                    {
                        "mode", "route_chunk_size", "completed_chunks",
                        "resumed_chunks", "resume_dir", "progress",
                        "prediction_chunk_seconds", "timing_scope",
                        "resource_scope",
                    },
                    f"{context}.execution",
                )
                if method != "neuromlr_greedy":
                    raise SchemaError(
                        f"{context}: chunked quality execution is Greedy-only"
                    )
                _integer(
                    execution["route_chunk_size"],
                    f"{context}.execution.route_chunk_size",
                    minimum=1,
                )
                completed = _integer(
                    execution["completed_chunks"],
                    f"{context}.execution.completed_chunks",
                    minimum=1,
                )
                resumed = _integer(
                    execution["resumed_chunks"],
                    f"{context}.execution.resumed_chunks",
                )
                if resumed > completed:
                    raise SchemaError(f"{context}: resumed chunks exceed completed chunks")
                for key in (
                    "resume_dir", "progress", "timing_scope", "resource_scope"
                ):
                    _string(execution[key], f"{context}.execution.{key}")
                chunk_seconds = execution["prediction_chunk_seconds"]
                if not isinstance(chunk_seconds, list) or len(chunk_seconds) != completed:
                    raise SchemaError(
                        f"{context}.execution.prediction_chunk_seconds length differs"
                    )
                for index, seconds in enumerate(chunk_seconds):
                    if _number(
                        seconds,
                        f"{context}.execution.prediction_chunk_seconds[{index}]",
                    ) < 0:
                        raise SchemaError(f"{context}: chunk seconds must be nonnegative")
                if execution["timing_scope"] != "sum_of_atomic_chunk_measurements_across_sessions":
                    raise SchemaError(f"{context}: wrong chunk timing scope")
                if execution["resource_scope"] != (
                    "maximum_observed_across_committed_sessions"
                ):
                    raise SchemaError(f"{context}: wrong chunk resource scope")
                if obj["warmup_repetitions"] != 0 or obj["measured_repetitions"] != 1:
                    raise SchemaError(
                        f"{context}: chunked quality prediction requires 0 warm-ups and 1 repetition"
                    )
                if obj["peak_cuda_memory_bytes"] <= 0:
                    raise SchemaError(
                        f"{context}: chunked NeuroMLR-G must report CUDA allocation"
                    )
            elif mode == "legacy_full_batch":
                execution = _exact(
                    execution, {"mode", "route_chunk_size"}, f"{context}.execution"
                )
                if execution["route_chunk_size"] != 0:
                    raise SchemaError(f"{context}: legacy route_chunk_size must be zero")
            else:
                raise SchemaError(f"{context}.execution.mode is unsupported")
        timing = _validate_prediction_timing(obj, timing_keys, context)
        warmups = _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
        measured = _integer(obj["measured_repetitions"], f"{context}.measured_repetitions", minimum=1)
        _validate_repetitions(timing, warmups, measured, "warmup_repetition_seconds", "prediction_repetition_seconds", f"{context}.timing")
        if "endpoint_failures" in obj:
            endpoint_failures = _integer(
                obj["endpoint_failures"], f"{context}.endpoint_failures"
            )
            if endpoint_failures > _integer(
                obj["samples"], f"{context}.samples", minimum=1
            ):
                raise SchemaError(f"{context}.endpoint_failures exceeds samples")
    elif schema in {
        "ewr.drncs-lg-prediction-diagnostics/v1",
        "ewr.drncs-lg-prediction-diagnostics/v2",
    }:
        v1_keys = {
            "schema", "method", "adapter_version", "audited_upstream_commit",
            "checkpoint", "checkpoint_sha256", "dataset_manifest",
            "dataset_manifest_sha256", "dataset_id", "network_id", "samples",
            "query_protocol", "endpoint_repair", "truth_interior_read_during_prediction",
            "endpoint_failures",
            "timing", "warmup_repetitions", "measured_repetitions", "seed", "workers",
            "inference_batch_size", "max_steps", "peak_rss_kib",
            "peak_cuda_memory_bytes", "environment",
        }
        timing_keys = {
            "data_and_graph_seconds", "model_load_seconds", "warmup_repetition_seconds",
            "prediction_repetition_seconds", "mean_prediction_seconds",
            "mean_seconds_per_query", "queries_per_second", "single_query_latency_samples",
            "single_query_latency_p50_seconds", "single_query_latency_p95_seconds",
            "single_query_latency_max_seconds", "component_stats_per_repetition",
            "total_process_seconds",
        }
        if schema.endswith("/v2"):
            keys = (v1_keys - {"truth_interior_read_during_prediction"}) | {
                "source",
                "dataset_split_role",
                "truth_interior_used_for_route_generation",
                "manifest_hash_pin_enforced",
                "records_hash_pin_enforced",
                "dataset_hash_pins_enforced",
                "generated_route_validity",
            }
            obj = _exact(value, keys, context)
            source = _object(obj["source"], f"{context}.source")
            _sha256_string(
                source.get("adapter_sha256"),
                f"{context}.source.adapter_sha256",
            )
            if obj["dataset_split_role"] not in {"validation", "test"}:
                raise SchemaError(
                    f"{context}.dataset_split_role must be validation or test"
                )
            for key in (
                "endpoint_repair",
                "truth_interior_used_for_route_generation",
                "manifest_hash_pin_enforced",
                "records_hash_pin_enforced",
                "dataset_hash_pins_enforced",
            ):
                _boolean(obj[key], f"{context}.{key}")
            validity_keys = {
                "queries",
                "empty_routes",
                "source_mismatches",
                "destination_mismatches",
                "routes_with_illegal_transitions",
                "illegal_transition_count",
                "non_simple_routes",
                "repeated_state_events",
            }
            validity = _exact(
                obj["generated_route_validity"],
                validity_keys,
                f"{context}.generated_route_validity",
            )
            for key in validity_keys:
                _integer(validity[key], f"{context}.generated_route_validity.{key}")
            if validity["destination_mismatches"] != obj["endpoint_failures"]:
                raise SchemaError(
                    f"{context}: generated-route destination mismatches differ from endpoint_failures"
                )
        else:
            obj = _exact(value, v1_keys, context)
        _schema(obj, schema, context)
        if method != "drncs_lg" or obj["method"] != method:
            raise SchemaError(f"{context}.method does not match {method!r}")
        _integer(obj["workers"], f"{context}.workers", minimum=1)
        _integer(obj["inference_batch_size"], f"{context}.inference_batch_size", minimum=1)
        endpoint_failures = _integer(
            obj["endpoint_failures"], f"{context}.endpoint_failures"
        )
        if endpoint_failures > _integer(
            obj["samples"], f"{context}.samples", minimum=1
        ):
            raise SchemaError(f"{context}.endpoint_failures exceeds samples")
        environment = _object(obj["environment"], f"{context}.environment")
        _string(environment.get("device"), f"{context}.environment.device")
        timing = _validate_prediction_timing(obj, timing_keys, context)
        warmups = _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
        measured = _integer(obj["measured_repetitions"], f"{context}.measured_repetitions", minimum=1)
        _validate_repetitions(timing, warmups, measured, "warmup_repetition_seconds", "prediction_repetition_seconds", f"{context}.timing")
    elif schema in {
        "ewr.drpk-static-diagnostics/v1", "ewr.drpk-static-diagnostics/v2"
    }:
        v1_keys = {
            "schema", "method", "provenance", "adaptation", "dataset_id",
            "dataset_manifest_sha256", "samples", "query_protocol", "truth_repair",
            "endpoint_failures", "timing", "peak_rss_kib", "peak_cuda_memory_bytes",
            "seed", "workers", "inference_batch_size", "warmup_repetitions",
            "measured_repetitions",
        }
        timing_keys = {
            "artifact_and_model_load_seconds", "warmup_repetition_seconds",
            "prediction_repetition_seconds", "warmup_component_totals_per_repetition",
            "component_totals_per_repetition", "component_totals", "prediction_seconds",
            "mean_seconds_per_query", "queries_per_second", "total_process_seconds",
        }
        if schema.endswith("/v1"):
            obj = _exact(value, v1_keys, context)
        else:
            obj = _exact(
                value,
                v1_keys | {
                    "resolved_device", "environment",
                    "preprocess_configuration_sha256", "checkpoint",
                },
                context,
                {"requested_device", "source"},
            )
            if "requested_device" in obj:
                _string(obj["requested_device"], f"{context}.requested_device")
            if "source" in obj:
                source = _object(obj["source"], f"{context}.source")
                _sha256_string(
                    source.get("adapter_sha256"),
                    f"{context}.source.adapter_sha256",
                )
            _string(obj["resolved_device"], f"{context}.resolved_device")
            environment = _object(obj["environment"], f"{context}.environment")
            environment_device = _string(
                environment.get("device"), f"{context}.environment.device"
            )
            if environment_device != obj["resolved_device"]:
                raise SchemaError(
                    f"{context}: resolved_device and environment.device differ"
                )
            _sha256_string(
                obj["preprocess_configuration_sha256"],
                f"{context}.preprocess_configuration_sha256",
            )
            if obj["checkpoint"] is not None:
                checkpoint = _exact(
                    obj["checkpoint"], {"path", "sha256", "epoch"},
                    f"{context}.checkpoint",
                )
                _string(checkpoint["path"], f"{context}.checkpoint.path")
                _sha256_string(
                    checkpoint["sha256"], f"{context}.checkpoint.sha256"
                )
                _integer(checkpoint["epoch"], f"{context}.checkpoint.epoch")
        _schema(obj, schema, context)
        if method not in {"drpk_static", "drp_tp"} or obj["method"] != method:
            raise SchemaError(f"{context}.method does not match {method!r}")
        _integer(obj["workers"], f"{context}.workers", minimum=1)
        _integer(obj["inference_batch_size"], f"{context}.inference_batch_size", minimum=1)
        endpoint_failures = _integer(
            obj["endpoint_failures"], f"{context}.endpoint_failures"
        )
        if endpoint_failures > _integer(obj["samples"], f"{context}.samples", minimum=1):
            raise SchemaError(f"{context}.endpoint_failures exceeds samples")
        timing = _validate_prediction_timing(obj, timing_keys, context)
        warmups = _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
        measured = _integer(obj["measured_repetitions"], f"{context}.measured_repetitions", minimum=1)
        _validate_repetitions(timing, warmups, measured, "warmup_repetition_seconds", "prediction_repetition_seconds", f"{context}.timing")
    else:
        raise SchemaError(f"{context}: unsupported prediction schema {schema!r}")

    for key in ("samples", "measured_repetitions"):
        _integer(obj[key], f"{context}.{key}", minimum=1)
    _integer(obj["warmup_repetitions"], f"{context}.warmup_repetitions")
    _nullable_integer(obj.get("peak_rss_kib"), f"{context}.peak_rss_kib")
    if "peak_cuda_memory_bytes" in obj:
        _integer(obj["peak_cuda_memory_bytes"], f"{context}.peak_cuda_memory_bytes")
    return obj


def validate_archive(value: dict[str, Any], context: str) -> dict[str, Any]:
    keys = {
        "schema_version", "status", "study", "repository", "scope", "protocol",
        "random_seed", "data", "quality", "training_runtime_supplementary",
        "oracle_efficiency", "quality_checks", "resume_candidates",
    }
    obj = _exact(value, keys, context)
    if _integer(obj["schema_version"], f"{context}.schema_version") != 1:
        raise SchemaError(f"{context}.schema_version must equal 1")
    if obj["status"] != "complete":
        raise SchemaError(f"{context}.status must be 'complete'")
    test = _object(_object(obj["quality"], f"{context}.quality").get("test"), f"{context}.quality.test")
    supplementary = _object(obj["training_runtime_supplementary"], f"{context}.training_runtime_supplementary")
    for name in ("project_edge_to_edge", "neuromlr_greedy"):
        entry = _object(test.get(name), f"{context}.quality.test.{name}")
        metrics = _object(entry.get("metrics"), f"{context}.quality.test.{name}.metrics")
        for key in METRIC_KEYS:
            metric = _number(metrics.get(key), f"{context}.quality.test.{name}.metrics.{key}")
            if metric > 1.0:
                raise SchemaError(f"{context}.quality.test.{name}.metrics.{key} must be <= 1")
        samples = _integer(
            metrics.get("samples"),
            f"{context}.quality.test.{name}.metrics.samples",
            minimum=1,
        )
        mismatches = _integer(
            entry.get("endpoint_mismatches"),
            f"{context}.quality.test.{name}.endpoint_mismatches",
        )
        if mismatches > samples:
            raise SchemaError(
                f"{context}.quality.test.{name}.endpoint_mismatches exceeds samples"
            )
    for name in (
        "project_full_common_train_500_updates", "neuromlr_50_epochs",
    ):
        _object(supplementary.get(name), f"{context}.training_runtime_supplementary.{name}")
    return obj


def _empty_quality() -> dict[str, Any]:
    return {
        "status": "pending", "sample_count": None, "edge_precision": None,
        "edge_recall": None, "edge_f1": None, "edge_jaccard": None,
        "exact_match": None,
        # endpoint_mismatches is retained as a v1 compatibility alias.
        "endpoint_failures": None, "endpoint_mismatches": None,
    }


def _empty_efficiency() -> dict[str, Any]:
    return {
        "status": "pending", "offline_seconds": None,
        "training_total_seconds": None, "threads": None, "device": None,
        "offline_device": None, "training_device": None,
        "prediction_device": None,
        "batch_boundary": None, "mean_batch_seconds": None,
        "mean_ms_per_query": None, "queries_per_second": None,
        "offline_peak_rss_kib": None, "training_peak_rss_kib": None,
        "prediction_peak_rss_kib": None,
        "training_peak_gpu_memory_bytes": None,
        "prediction_peak_gpu_memory_bytes": None,
        "operational_full_test_samples": None,
        "operational_timing_complete": None,
        "operational_known_active_wall_lower_bound_seconds": None,
        "operational_wall_seconds": None,
        "operational_successful_final_attempt_wall_seconds": None,
        "operational_wasted_interrupted_wall_seconds": None,
        "operational_attempt_count": None,
        "operational_lost_attempt_count": None,
        "internal_prediction_seconds": None,
        "shard_adapter_process_seconds": None,
        "operational_time_report_sha256": None,
        "operational_comparability_note": None,
    }


def _empty_artifacts() -> dict[str, Any]:
    return {
        "status": "pending", "items": [], "model_bytes": None,
        "auxiliary_bytes": None, "model_and_auxiliary_bytes": None,
    }


def _set_endpoint_failures(quality: dict[str, Any], value: int) -> None:
    """Set the canonical endpoint failure count and its v1 alias together."""

    quality["endpoint_failures"] = value
    quality["endpoint_mismatches"] = value


def _quality_status(section: Mapping[str, Any]) -> str:
    required = (
        "sample_count", "edge_precision", "edge_recall", "edge_f1",
        "edge_jaccard", "exact_match", "endpoint_failures",
    )
    present = sum(section[key] is not None for key in required)
    if present == 0:
        return "pending"
    return "complete" if present == len(required) else "partial"


def _uses_cuda(device: Any) -> bool:
    return device is not None and "cuda" in str(device).lower()


def _efficiency_status(method: str, section: Mapping[str, Any]) -> str:
    values = [value for key, value in section.items() if key != "status"]
    if all(value is None for value in values):
        return "pending"

    phases = EFFICIENCY_PHASES[method]
    for phase in phases:
        if phase == "offline":
            required = (
                "offline_seconds", "offline_peak_rss_kib", "offline_device"
            )
        elif phase == "training":
            required = (
                "training_total_seconds", "training_peak_rss_kib",
                "training_device",
            )
        else:
            required = (
                "prediction_device", "batch_boundary", "mean_batch_seconds",
                "mean_ms_per_query", "queries_per_second",
                "prediction_peak_rss_kib",
            )
            if method in INFERENCE_THREADS_APPLICABLE:
                required += ("threads",)
        if any(section[key] is None for key in required):
            return "partial"

    if "training" in phases and _uses_cuda(section["training_device"]):
        if section["training_peak_gpu_memory_bytes"] is None:
            return "partial"
    if "prediction" in phases and _uses_cuda(section["prediction_device"]):
        if section["prediction_peak_gpu_memory_bytes"] is None:
            return "partial"
    return "complete"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_string(value: Any, context: str) -> str:
    result = _string(value, context).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{context} must be a 64-character hexadecimal SHA-256")
    return result


def _register_artifact(
    section: dict[str, Any], *, kind: str, role: str, declared_path: str | None,
    declared_sha256: str | None, source: str, measured_path: Path | None = None,
) -> None:
    if kind not in ARTIFACT_KINDS:
        raise SchemaError(f"artifact kind must be one of {sorted(ARTIFACT_KINDS)}")
    role = _string(role, "artifact role")
    if measured_path is not None:
        measured_path = measured_path.resolve()
        if not measured_path.is_file():
            raise SchemaError(f"declared artifact does not exist: {measured_path}")
        path_text = str(measured_path)
        byte_count: int | None = measured_path.stat().st_size
        observed_sha256: str | None = _sha256_file(measured_path)
        if declared_sha256 is not None and observed_sha256 != declared_sha256:
            raise SchemaError(
                f"artifact {role!r} SHA-256 differs from its declared identity"
            )
    else:
        # Diagnostics may bind a relative path in the producer's invocation
        # directory.  The summarizer records that declaration but never guesses
        # a new base directory in order to manufacture a byte count or hash.
        path_text = declared_path
        byte_count = None
        observed_sha256 = declared_sha256
        if declared_path is not None:
            absolute = Path(declared_path).expanduser()
            if absolute.is_absolute() and absolute.is_file():
                path_text = str(absolute.resolve())
                byte_count = absolute.stat().st_size
                computed = _sha256_file(absolute)
                if declared_sha256 is not None and computed != declared_sha256:
                    raise SchemaError(
                        f"artifact {role!r} SHA-256 differs from its declared identity"
                    )
                observed_sha256 = computed

    candidate = {
        "kind": kind, "role": role, "path": path_text, "bytes": byte_count,
        "sha256": observed_sha256, "sources": [source],
    }
    existing = next(
        (item for item in section["items"] if item["role"] == role), None
    )
    if existing is None:
        section["items"].append(candidate)
        return
    if existing["kind"] != kind:
        raise SchemaError(f"artifact role {role!r} has conflicting kinds")
    for key in ("path", "bytes", "sha256"):
        old, new = existing[key], candidate[key]
        if old is not None and new is not None and old != new:
            if key == "path" and existing["bytes"] is not None and candidate["bytes"] is None:
                continue
            if key == "path" and existing["bytes"] is None and candidate["bytes"] is not None:
                existing[key] = new
                continue
            raise SchemaError(f"artifact role {role!r} has conflicting {key}")
        if old is None:
            existing[key] = new
    if source not in existing["sources"]:
        existing["sources"].append(source)


def _finalize_artifacts(method: str, section: dict[str, Any]) -> None:
    items = section["items"]
    if not items:
        section["status"] = "pending"
        return
    required_kinds = set()
    if method in MODEL_ARTIFACT_METHODS:
        required_kinds.add("model")
    if method in AUXILIARY_ARTIFACT_METHODS:
        required_kinds.add("auxiliary")
    complete = all(
        any(item["kind"] == required_kind for item in items)
        for required_kind in required_kinds
    )
    complete = complete and all(
        item["bytes"] is not None and item["sha256"] is not None for item in items
    )
    for kind, output_key in (
        ("model", "model_bytes"), ("auxiliary", "auxiliary_bytes")
    ):
        selected = [item for item in items if item["kind"] == kind]
        if not selected:
            section[output_key] = 0 if kind not in required_kinds else None
        elif all(item["bytes"] is not None for item in selected):
            section[output_key] = sum(item["bytes"] for item in selected)
    if all(item["bytes"] is not None for item in items):
        section["model_and_auxiliary_bytes"] = sum(
            item["bytes"] for item in items
        )
    section["status"] = "complete" if complete else "partial"


def _set_phase_device(efficiency: dict[str, Any], phase: str, device: Any) -> None:
    if device is None:
        return
    normalized = _string(device, f"{phase} device")
    key = f"{phase}_device"
    current = efficiency[key]
    if current is not None and current != normalized:
        raise SchemaError(
            f"conflicting {phase} devices {current!r} and {normalized!r}"
        )
    efficiency[key] = normalized


def _finalize_device(efficiency: dict[str, Any]) -> None:
    # `device` remains the v1 compatibility field and denotes the measured
    # prediction device when available.  Phase-specific fields remove ambiguity
    # for CPU preprocessing followed by GPU training/inference.
    efficiency["device"] = (
        efficiency["prediction_device"]
        or efficiency["training_device"]
        or efficiency["offline_device"]
    )


def _device_from_diagnostics(obj: Mapping[str, Any]) -> str | None:
    environment = obj.get("environment")
    if isinstance(environment, dict) and isinstance(environment.get("device"), str):
        return environment["device"]
    peak_cuda = obj.get("peak_cuda_memory_bytes")
    if isinstance(peak_cuda, int) and not isinstance(peak_cuda, bool):
        return "cuda" if peak_cuda > 0 else "cpu"
    return None


def _apply_archive(
    methods: dict[str, dict[str, Any]], archive: dict[str, Any], source: str
) -> None:
    test = archive["quality"]["test"]
    supplementary = archive["training_runtime_supplementary"]
    for method, archive_name in (
        ("project", "project_edge_to_edge"),
        ("neuromlr_greedy", "neuromlr_greedy"),
    ):
        row = methods[method]
        if row["sources"]["evaluation"] is None:
            entry = test[archive_name]
            metrics = entry["metrics"]
            row["quality"].update(
                sample_count=metrics["samples"],
                edge_precision=metrics["edge_precision"],
                edge_recall=metrics["edge_recall"], edge_f1=metrics["edge_f1"],
                edge_jaccard=metrics["edge_jaccard"], exact_match=metrics["exact_match"],
            )
            _set_endpoint_failures(row["quality"], entry["endpoint_mismatches"])
            row["sources"]["archive"] = source

    selection = archive["quality"].get("validation_selection")
    if isinstance(selection, dict):
        for methods_for_checkpoint, selection_name in (
            (("project",), "project"),
            (("neuromlr_greedy",), "neuromlr_greedy"),
        ):
            selected = selection.get(selection_name)
            if not isinstance(selected, dict) or "checkpoint_sha256" not in selected:
                continue
            checkpoint_sha256 = _sha256_string(
                selected["checkpoint_sha256"],
                f"archive quality.validation_selection.{selection_name}.checkpoint_sha256",
            )
            for method in methods_for_checkpoint:
                # A formal input manifest is authoritative for the run being
                # summarized.  The archive is only a fallback and may describe
                # an older byte-level checkpoint identity even when the same
                # validation update/epoch was selected again.
                if any(
                    artifact["role"] == "selected_checkpoint"
                    for artifact in methods[method]["artifacts"]["items"]
                ):
                    continue
                _register_artifact(
                    methods[method]["artifacts"], kind="model",
                    role="selected_checkpoint", declared_path=None,
                    declared_sha256=checkpoint_sha256, source=source,
                )

    project = methods["project"]
    if project["sources"]["training"] is None:
        training = supplementary["project_full_common_train_500_updates"]
        project["efficiency"].update(
            training_total_seconds=_number(training["wall_seconds"], "archive project wall_seconds"),
            threads=_integer(training["threads"], "archive project threads", minimum=1),
            training_peak_rss_kib=_integer(training["peak_rss_kib"], "archive project peak_rss_kib"),
        )
        _set_phase_device(project["efficiency"], "training", "cpu")
        project["sources"]["archive"] = source

    neuro = methods["neuromlr_greedy"]
    if neuro["sources"]["training"] is None:
        training = supplementary["neuromlr_50_epochs"]
        neuro["efficiency"].update(
            training_total_seconds=_number(
                training["total_seconds"], "archive NeuroMLR total_seconds"
            ),
            training_peak_rss_kib=_integer(
                training["peak_rss_kib"], "archive NeuroMLR peak_rss_kib"
            ),
        )
        _set_phase_device(
            neuro["efficiency"], "training",
            _string(training["device"], "archive NeuroMLR device"),
        )
        neuro["sources"]["archive"] = source


def _apply_training(method: str, obj: dict[str, Any], efficiency: dict[str, Any]) -> None:
    schema = obj.get("schema")
    if schema == "ewr.static-route-baseline-training-diagnostics/v1":
        total = float(obj["timing"]["total_before_artifact_write_seconds"])
        efficiency["threads"] = obj["threads"]
        if method == "sp_length":
            efficiency["offline_seconds"] = total
            efficiency["offline_peak_rss_kib"] = obj["peak_rss_kib"]
            _set_phase_device(efficiency, "offline", "cpu")
        else:
            efficiency["training_total_seconds"] = total
            efficiency["training_peak_rss_kib"] = obj["peak_rss_kib"]
            _set_phase_device(efficiency, "training", "cpu")
    elif schema == "ewr.project-training-summary/v1":
        efficiency.update(
            training_total_seconds=obj["timing"]["total_before_summary_write_seconds"],
            threads=obj["threads"],
            training_peak_rss_kib=obj["peak_rss_kib"],
        )
        _set_phase_device(efficiency, "training", "cpu")
    elif schema in {
        "ewr.drncs-lg-training-diagnostics/v1",
        "ewr.drncs-lg-training-diagnostics/v2",
    }:
        configuration = obj["configuration"]
        efficiency.update(
            training_total_seconds=obj["total_process_seconds"],
            threads=configuration.get("workers"),
            training_peak_rss_kib=obj["peak_rss_kib"],
            training_peak_gpu_memory_bytes=obj["peak_cuda_memory_bytes"],
        )
        _set_phase_device(
            efficiency, "training", obj["environment"].get("device")
        )
    elif schema in {"ewr.drpk-static-selection/v1", "ewr.drpk-static-selection/v2"}:
        efficiency.update(
            training_total_seconds=obj["total_seconds"],
            training_peak_rss_kib=obj["peak_rss_kib"],
            training_peak_gpu_memory_bytes=obj.get("peak_cuda_memory_bytes"),
        )
        if "workers" in obj:
            efficiency["threads"] = obj["workers"]
        _set_phase_device(efficiency, "training", _device_from_diagnostics(obj))
    else:  # legacy NeuroMLR selection
        efficiency.update(
            training_total_seconds=obj["total_seconds"],
            training_peak_rss_kib=obj["peak_rss_kib"],
        )


def _apply_offline(method: str, obj: dict[str, Any], efficiency: dict[str, Any]) -> None:
    if obj["schema"] in {
        "ewr.drncs-lg-preprocess/v1",
        "ewr.drncs-lg-preprocess/v2",
    }:
        efficiency.update(
            offline_seconds=obj["timing"]["total_process_seconds"],
            offline_peak_rss_kib=obj["peak_rss_kib"],
            threads=obj["configuration"].get("workers"),
        )
        _set_phase_device(
            efficiency, "offline", obj["environment"].get("device")
        )
    else:
        if method == "drp_tp" and "drp_tp_ready_seconds" in obj["timing"]:
            offline_seconds = obj["timing"]["drp_tp_ready_seconds"]
            offline_peak_rss_kib = obj["timing"]["drp_tp_ready_peak_rss_kib"]
        else:
            offline_seconds = obj["timing"]["total_seconds"]
            offline_peak_rss_kib = obj["peak_rss_kib"]
        efficiency.update(
            offline_seconds=offline_seconds,
            offline_peak_rss_kib=offline_peak_rss_kib,
            threads=obj["configuration"].get("workers"),
        )
        _set_phase_device(efficiency, "offline", "cpu")


def _apply_prediction(
    method: str, obj: dict[str, Any], quality: dict[str, Any], efficiency: dict[str, Any]
) -> None:
    schema = obj["schema"]
    samples = obj["samples"]
    if quality["sample_count"] is None:
        quality["sample_count"] = samples
    quality_only = schema == SHARDED_QUALITY_PREDICTION_SCHEMA or (
        schema == "ewr.neuromlr-diagnostics/v1"
        and isinstance(obj.get("execution"), dict)
        and obj["execution"].get("mode") == NEUROMLR_CHUNKED_QUALITY_MODE
    )
    if quality_only:
        _set_endpoint_failures(quality, obj["endpoint_failures"])
        return
    timing = obj["timing"]
    warmups, measured = obj["warmup_repetitions"], obj["measured_repetitions"]
    efficiency.update(
        mean_ms_per_query=1000.0 * timing["mean_seconds_per_query"],
        queries_per_second=timing["queries_per_second"],
        prediction_peak_rss_kib=obj.get("peak_rss_kib"),
        prediction_peak_gpu_memory_bytes=obj.get("peak_cuda_memory_bytes"),
    )
    if schema in {
        "ewr.static-route-baseline-prediction-diagnostics/v1",
        "ewr.project-prediction-diagnostics/v1",
    }:
        efficiency.update(
            threads=obj["threads"],
            batch_boundary=timing["timing_boundary"],
            mean_batch_seconds=timing["mean_metric_and_query_seconds"],
        )
        _set_phase_device(efficiency, "prediction", "cpu")
        if schema == "ewr.project-prediction-diagnostics/v1":
            # The edge-to-edge CCH query is constructed with the true final
            # raw edge and only emits a successful complete route.
            _set_endpoint_failures(quality, 0)
    elif schema == "ewr.neuromlr-diagnostics/v1":
        efficiency.update(
            batch_boundary=f"full ordered batch of {samples} queries; {warmups} warm-up + {measured} measured repetitions",
            mean_batch_seconds=timing["prediction_seconds"],
        )
        _set_phase_device(
            efficiency, "prediction", _device_from_diagnostics(obj)
        )
        if "endpoint_failures" in obj:
            _set_endpoint_failures(quality, obj["endpoint_failures"])
    elif schema in {
        "ewr.drncs-lg-prediction-diagnostics/v1",
        "ewr.drncs-lg-prediction-diagnostics/v2",
    }:
        efficiency.update(
            threads=obj["workers"],
            batch_boundary=(
                f"full ordered batch of {samples} queries (rollout minibatch {obj['inference_batch_size']}); "
                f"{warmups} warm-up + {measured} measured repetitions"
            ),
            mean_batch_seconds=timing["mean_prediction_seconds"],
        )
        _set_phase_device(
            efficiency, "prediction", obj["environment"].get("device")
        )
        _set_endpoint_failures(quality, obj["endpoint_failures"])
    else:
        efficiency.update(
            threads=obj["workers"],
            batch_boundary=(
                f"full ordered batch of {samples} queries (key-model minibatch {obj['inference_batch_size']}); "
                f"{warmups} warm-up + {measured} measured repetitions"
            ),
            mean_batch_seconds=timing["prediction_seconds"],
        )
        _set_phase_device(
            efficiency, "prediction", _device_from_diagnostics(obj)
        )
        _set_endpoint_failures(quality, obj["endpoint_failures"])
    if "endpoint_mismatches" in obj:
        _set_endpoint_failures(quality, obj["endpoint_mismatches"])


def validate_operational_prediction(
    value: dict[str, Any], method: str, context: str
) -> dict[str, Any]:
    obj = _exact(
        value,
        {"schema", "samples", "methods", "comparability_note"},
        context,
    )
    _schema(obj, OPERATIONAL_EFFICIENCY_SCHEMA, context)
    samples = _integer(obj["samples"], f"{context}.samples", minimum=1)
    _string(obj["comparability_note"], f"{context}.comparability_note")
    methods = _object(obj["methods"], f"{context}.methods")
    expected_methods = {
        "project", "sp_length", "markov_sp", "neuromlr_greedy", "drncs_lg",
        "drpk_static", "drp_tp",
    }
    if set(methods) != expected_methods:
        raise SchemaError(f"{context}.methods differs from the seven-method protocol")
    entry = _exact(
        methods.get(method),
        {
            "samples", "outer_boundary", "wall_seconds",
            "timing_complete", "known_active_wall_lower_bound_seconds",
            "successful_final_attempt_wall_seconds",
            "wasted_interrupted_wall_seconds", "attempt_count",
            "lost_attempt_count",
            "mean_ms_per_query", "queries_per_second", "user_seconds",
            "system_seconds", "peak_rss_kib", "exit_status", "device",
            "peak_cuda_memory_bytes", "internal_prediction_seconds",
            "shard_adapter_process_seconds", "time_evidence", "diagnostic",
        },
        f"{context}.methods.{method}",
    )
    if _integer(
        entry["samples"], f"{context}.methods.{method}.samples", minimum=1
    ) != samples:
        raise SchemaError(f"{context}: operational sample counts differ")
    timing_complete = _boolean(
        entry["timing_complete"], f"{context}.{method}.timing_complete"
    )
    lower_bound = _number(
        entry["known_active_wall_lower_bound_seconds"],
        f"{context}.{method}.known_active_wall_lower_bound_seconds",
    )
    wall = (
        _number(entry["wall_seconds"], f"{context}.{method}.wall_seconds")
        if entry["wall_seconds"] is not None
        else None
    )
    final = (
        _number(
            entry["successful_final_attempt_wall_seconds"],
            f"{context}.{method}.successful_final_attempt_wall_seconds",
        )
        if entry["successful_final_attempt_wall_seconds"] is not None
        else None
    )
    wasted = _number(
        entry["wasted_interrupted_wall_seconds"],
        f"{context}.{method}.wasted_interrupted_wall_seconds",
    )
    attempts = _integer(
        entry["attempt_count"], f"{context}.{method}.attempt_count", minimum=1
    )
    lost = _integer(
        entry["lost_attempt_count"], f"{context}.{method}.lost_attempt_count"
    )
    if lost > attempts or lower_bound < 0 or wasted < 0:
        raise SchemaError(f"{context}.{method}: invalid lost-attempt accounting")
    if timing_complete:
        if lost != 0 or wall is None or final is None or wall <= 0 or final <= 0:
            raise SchemaError(f"{context}.{method}: complete timing fields are missing")
        if not math.isclose(wall, final + wasted, rel_tol=0.0, abs_tol=1e-9):
            raise SchemaError(f"{context}.{method}: attempt wall-time decomposition differs")
        if not math.isclose(lower_bound, wall, rel_tol=0.0, abs_tol=1e-9):
            raise SchemaError(f"{context}.{method}: complete timing lower bound differs")
        if entry["exit_status"] != 0:
            raise SchemaError(f"{context}.{method}: final exit status is not zero")
        expected_ms = 1000.0 * wall / samples
        expected_qps = samples / wall
        if not math.isclose(
            _number(entry["mean_ms_per_query"], f"{context}.{method}.mean_ms_per_query"),
            expected_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            _number(entry["queries_per_second"], f"{context}.{method}.queries_per_second"),
            expected_qps,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SchemaError(f"{context}.{method}: operational rate differs from wall time")
    elif lost == 0 or wall is not None or entry["mean_ms_per_query"] is not None or entry["queries_per_second"] is not None:
        raise SchemaError(f"{context}.{method}: incomplete timing claims an exact rate")
    for key in ("user_seconds", "system_seconds", "internal_prediction_seconds"):
        if _number(entry[key], f"{context}.{method}.{key}") < 0:
            raise SchemaError(f"{context}.{method}.{key} must be nonnegative")
    if entry["peak_rss_kib"] is not None:
        _integer(entry["peak_rss_kib"], f"{context}.{method}.peak_rss_kib", minimum=1)
    peak_cuda = _integer(
        entry["peak_cuda_memory_bytes"],
        f"{context}.{method}.peak_cuda_memory_bytes",
    )
    device = _string(entry["device"], f"{context}.{method}.device")
    if method in {"neuromlr_greedy", "drncs_lg", "drpk_static"}:
        if "cuda" not in device.lower() or peak_cuda <= 0:
            raise SchemaError(f"{context}.{method}: CUDA evidence is missing")
    elif device != "cpu" or peak_cuda != 0:
        raise SchemaError(f"{context}.{method}: CPU-only device evidence differs")
    shard_process = entry["shard_adapter_process_seconds"]
    if method in {"drncs_lg", "drpk_static", "drp_tp"}:
        if shard_process is None or _number(
            shard_process, f"{context}.{method}.shard_adapter_process_seconds"
        ) <= 0:
            raise SchemaError(f"{context}.{method}: shard process decomposition is missing")
    elif shard_process is not None:
        raise SchemaError(f"{context}.{method}: direct method declares shard process time")
    for key in ("time_evidence", "diagnostic"):
        evidence = _object(entry[key], f"{context}.{method}.{key}")
        required = {"path", "sha256"} | ({"schema"} if key == "diagnostic" else set())
        if set(evidence) != required:
            raise SchemaError(f"{context}.{method}.{key} shape differs")
        _string(evidence["path"], f"{context}.{method}.{key}.path")
        _sha256_string(evidence["sha256"], f"{context}.{method}.{key}.sha256")
        if key == "diagnostic":
            _string(evidence["schema"], f"{context}.{method}.{key}.schema")
    _string(entry["outer_boundary"], f"{context}.{method}.outer_boundary")
    return entry


def _apply_operational_prediction(
    entry: Mapping[str, Any], note: str, efficiency: dict[str, Any]
) -> None:
    efficiency.update(
        operational_full_test_samples=entry["samples"],
        operational_timing_complete=entry["timing_complete"],
        operational_known_active_wall_lower_bound_seconds=entry[
            "known_active_wall_lower_bound_seconds"
        ],
        operational_wall_seconds=entry["wall_seconds"],
        operational_successful_final_attempt_wall_seconds=entry[
            "successful_final_attempt_wall_seconds"
        ],
        operational_wasted_interrupted_wall_seconds=entry[
            "wasted_interrupted_wall_seconds"
        ],
        operational_attempt_count=entry["attempt_count"],
        operational_lost_attempt_count=entry["lost_attempt_count"],
        mean_batch_seconds=entry["wall_seconds"],
        mean_ms_per_query=entry["mean_ms_per_query"],
        queries_per_second=entry["queries_per_second"],
        prediction_peak_rss_kib=entry["peak_rss_kib"],
        prediction_peak_gpu_memory_bytes=entry["peak_cuda_memory_bytes"],
        internal_prediction_seconds=entry["internal_prediction_seconds"],
        shard_adapter_process_seconds=entry["shard_adapter_process_seconds"],
        operational_time_report_sha256=entry["time_evidence"]["sha256"],
        operational_comparability_note=note,
        batch_boundary=entry["outer_boundary"],
    )
    _set_phase_device(efficiency, "prediction", entry["device"])


def _collect_bound_artifacts(
    method: str, phase: str, obj: Mapping[str, Any], source: Path,
    section: dict[str, Any],
) -> None:
    """Collect only files explicitly bound by a validated diagnostic."""

    source_text = str(source)
    if phase == "offline" and obj.get("schema") in {
        "ewr.drncs-lg-preprocess/v1",
        "ewr.drncs-lg-preprocess/v2",
    }:
        _register_artifact(
            section, kind="auxiliary", role="preprocess_artifact",
            declared_path=_string(obj["artifact"], f"{source}.artifact"),
            declared_sha256=_sha256_string(
                obj["artifact_sha256"], f"{source}.artifact_sha256"
            ),
            source=source_text,
        )
        return

    checkpoint: Any = None
    checkpoint_sha256: Any = None
    if phase == "training" and obj.get("schema") in {
        "ewr.drncs-lg-training-diagnostics/v1",
        "ewr.drncs-lg-training-diagnostics/v2",
    }:
        checkpoint = obj.get("checkpoint")
        checkpoint_sha256 = obj.get("checkpoint_sha256")
    elif phase == "training" and obj.get("schema") in {
        "ewr.drpk-static-selection/v1", "ewr.drpk-static-selection/v2"
    }:
        selected = obj.get("selected")
        if isinstance(selected, dict):
            checkpoint = selected.get("checkpoint")
            checkpoint_sha256 = selected.get("checkpoint_sha256")
    elif phase == "training" and "schema_version" in obj and method == "neuromlr_greedy":
        selected = obj.get("selected")
        if isinstance(selected, dict):
            checkpoint = selected.get("checkpoint")
            checkpoint_sha256 = selected.get("checkpoint_sha256")
    elif phase == "prediction" and obj.get("schema") in {
        "ewr.neuromlr-diagnostics/v1", "ewr.drncs-lg-prediction-diagnostics/v1",
        "ewr.drncs-lg-prediction-diagnostics/v2",
        "ewr.drpk-static-diagnostics/v1", "ewr.drpk-static-diagnostics/v2",
    }:
        checkpoint = obj.get("checkpoint")
        checkpoint_sha256 = obj.get("checkpoint_sha256")

    if isinstance(checkpoint, dict):
        checkpoint_sha256 = checkpoint.get("sha256")
        checkpoint = checkpoint.get("path")

    if checkpoint is None:
        return
    declared_hash = (
        _sha256_string(checkpoint_sha256, f"{source}.checkpoint_sha256")
        if checkpoint_sha256 is not None else None
    )
    _register_artifact(
        section, kind="model", role="selected_checkpoint",
        declared_path=_string(checkpoint, f"{source}.checkpoint"),
        declared_sha256=declared_hash, source=source_text,
    )


def _resolve_path(raw: Any, base: Path, context: str) -> Path | None:
    if raw is None:
        return None
    path = Path(_string(raw, context)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise SchemaError(f"{context}: file does not exist: {path}")
    return path


def _resolve_artifact_specs(
    raw: Any, runtime_root: Path, context: str
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SchemaError(f"{context} must be an array")
    result = []
    roles: set[str] = set()
    for index, value in enumerate(raw):
        item_context = f"{context}[{index}]"
        item = _exact(value, {"kind", "role", "path"}, item_context, {"sha256"})
        kind = _string(item["kind"], f"{item_context}.kind")
        if kind not in ARTIFACT_KINDS:
            raise SchemaError(
                f"{item_context}.kind must be one of {sorted(ARTIFACT_KINDS)}"
            )
        role = _string(item["role"], f"{item_context}.role")
        if role in roles:
            raise SchemaError(f"{context} contains duplicate role {role!r}")
        roles.add(role)
        path = _resolve_path(item["path"], runtime_root, f"{item_context}.path")
        assert path is not None
        expected_sha256 = (
            _sha256_string(item["sha256"], f"{item_context}.sha256")
            if "sha256" in item else None
        )
        result.append(
            {
                "kind": kind, "role": role, "path": path,
                "sha256": expected_sha256,
            }
        )
    return result


def _auto_paths(runtime_root: Path | None) -> dict[str, dict[str, Any]]:
    result = {method: {key: None for key in METHOD_INPUT_KEYS} for method in METHODS}
    if runtime_root is None:
        return result
    for method, kinds in AUTO_ARTIFACTS.items():
        for kind, candidates in kinds.items():
            matches = [runtime_root / candidate for candidate in candidates if (runtime_root / candidate).is_file()]
            if matches:
                result[method][kind] = matches[0].resolve()
    return result


def _load_manifest(
    path: Path | None, cli_runtime_root: Path | None
) -> tuple[Path | None, dict[str, dict[str, Any]], object, tuple[str, ...] | None]:
    unset = object()
    archive_raw: object = unset
    if path is None:
        runtime_root = cli_runtime_root.resolve() if cli_runtime_root else None
        if runtime_root is not None and not runtime_root.is_dir():
            raise SchemaError(f"runtime root does not exist: {runtime_root}")
        return runtime_root, _auto_paths(runtime_root), archive_raw, None
    manifest = load_json(path)
    manifest = _exact(
        manifest,
        {"schema", "methods"},
        str(path),
        {"runtime_root", "archived_summary", "included_methods"},
    )
    _schema(manifest, INPUT_SCHEMA, str(path))
    methods = _object(manifest["methods"], f"{path}.methods")
    unknown = sorted(methods.keys() - METHODS.keys())
    if unknown:
        raise SchemaError(f"{path}.methods: unknown methods {unknown}")
    included_methods: tuple[str, ...] | None = None
    if "included_methods" in manifest:
        raw_included = manifest["included_methods"]
        if (
            not isinstance(raw_included, list)
            or not raw_included
            or any(not isinstance(method, str) for method in raw_included)
        ):
            raise SchemaError(f"{path}.included_methods must be a non-empty string array")
        included_methods = tuple(raw_included)
        if len(set(included_methods)) != len(included_methods):
            raise SchemaError(f"{path}.included_methods contains duplicates")
        unknown_included = sorted(set(included_methods) - METHODS.keys())
        if unknown_included:
            raise SchemaError(
                f"{path}.included_methods: unknown methods {unknown_included}"
            )
        canonical_order = tuple(
            method for method in METHODS if method in set(included_methods)
        )
        if included_methods != canonical_order:
            raise SchemaError(
                f"{path}.included_methods must follow the registered method order"
            )
        omitted_entries = sorted(methods.keys() - set(included_methods))
        if omitted_entries:
            raise SchemaError(
                f"{path}.methods contains excluded entries {omitted_entries}"
            )
    raw_root = cli_runtime_root
    if raw_root is None and "runtime_root" in manifest:
        raw_root = Path(
            _string(manifest["runtime_root"], f"{path}.runtime_root")
        ).expanduser()
    if raw_root is None:
        raw_root = Path(".")
    runtime_root = (
        raw_root.resolve()
        if raw_root.is_absolute() or cli_runtime_root is not None
        else (path.parent / raw_root).resolve()
    )
    if not runtime_root.is_dir():
        raise SchemaError(f"runtime root does not exist: {runtime_root}")
    resolved = _auto_paths(runtime_root)
    for method, raw_entry in methods.items():
        entry = _exact(raw_entry, set(), f"{path}.methods.{method}", METHOD_INPUT_KEYS)
        for kind, raw_value in entry.items():
            if kind == "artifacts":
                resolved[method][kind] = _resolve_artifact_specs(
                    raw_value, runtime_root,
                    f"{path}.methods.{method}.artifacts",
                )
            else:
                resolved[method][kind] = _resolve_path(
                    raw_value, runtime_root, f"{path}.methods.{method}.{kind}"
                )
    if "archived_summary" in manifest:
        archive_raw = manifest["archived_summary"]
        if archive_raw is not None:
            archive_raw = _resolve_path(
                archive_raw, path.parent, f"{path}.archived_summary"
            )
    return runtime_root, resolved, archive_raw, included_methods


def summarize(
    paths: dict[str, dict[str, Any]],
    archive_path: Path | None,
    included_methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for method, label in METHODS.items():
        rows[method] = {
            "id": method, "label": label, "status": "pending",
            "quality": _empty_quality(), "efficiency": _empty_efficiency(),
            "artifacts": _empty_artifacts(),
            "sources": {
                "evaluation": None, "offline": None, "training": None,
                "prediction": None, "operational_prediction": None,
                "artifacts": [], "archive": None,
            },
        }
    for method in METHODS:
        row = rows[method]
        method_paths = paths.get(method, {})
        evaluation_path = method_paths.get("evaluation")
        if evaluation_path is not None:
            evaluated = validate_evaluation(load_json(evaluation_path), str(evaluation_path))
            row["quality"].update(evaluated)
            row["sources"]["evaluation"] = str(evaluation_path)
        offline_path = method_paths.get("offline")
        if offline_path is not None:
            offline = validate_offline(load_json(offline_path), method, str(offline_path))
            _apply_offline(method, offline, row["efficiency"])
            _collect_bound_artifacts(
                method, "offline", offline, offline_path, row["artifacts"]
            )
            row["sources"]["offline"] = str(offline_path)
        training_path = method_paths.get("training")
        if training_path is not None:
            training = validate_training(load_json(training_path), method, str(training_path))
            _apply_training(method, training, row["efficiency"])
            _collect_bound_artifacts(
                method, "training", training, training_path, row["artifacts"]
            )
            row["sources"]["training"] = str(training_path)
        prediction_path = method_paths.get("prediction")
        operational_prediction_path = method_paths.get("operational_prediction")
        if operational_prediction_path is not None and prediction_path is None:
            raise SchemaError(
                f"{method}: operational_prediction requires a quality prediction"
            )
        if prediction_path is not None:
            prediction = validate_prediction(load_json(prediction_path), method, str(prediction_path))
            if (
                row["quality"]["sample_count"] is not None
                and row["quality"]["sample_count"] != prediction["samples"]
            ):
                raise SchemaError(
                    f"{method}: evaluator sample_count and prediction samples differ"
                )
            _apply_prediction(
                method, prediction, row["quality"], row["efficiency"]
            )
            _collect_bound_artifacts(
                method, "prediction", prediction, prediction_path,
                row["artifacts"],
            )
            row["sources"]["prediction"] = str(prediction_path)
        if operational_prediction_path is not None:
            operational_document = load_json(operational_prediction_path)
            operational_entry = validate_operational_prediction(
                operational_document, method, str(operational_prediction_path)
            )
            if (
                row["quality"]["sample_count"] is not None
                and row["quality"]["sample_count"] != operational_entry["samples"]
            ):
                raise SchemaError(
                    f"{method}: evaluator and operational sample counts differ"
                )
            _apply_operational_prediction(
                operational_entry,
                operational_document["comparability_note"],
                row["efficiency"],
            )
            row["sources"]["operational_prediction"] = str(
                operational_prediction_path
            )
        for artifact in method_paths.get("artifacts") or []:
            _register_artifact(
                row["artifacts"], kind=artifact["kind"], role=artifact["role"],
                declared_path=str(artifact["path"]),
                declared_sha256=artifact["sha256"],
                measured_path=artifact["path"], source="input manifest",
            )
            row["sources"]["artifacts"].append(str(artifact["path"]))
    if archive_path is not None:
        archive = validate_archive(load_json(archive_path), str(archive_path))
        _apply_archive(rows, archive, str(archive_path))
    for method, row in rows.items():
        _finalize_device(row["efficiency"])
        _finalize_artifacts(method, row["artifacts"])
        row["quality"]["status"] = _quality_status(row["quality"])
        row["efficiency"]["status"] = _efficiency_status(
            method, row["efficiency"]
        )
        statuses = {row["quality"]["status"], row["efficiency"]["status"]}
        if statuses == {"pending"}:
            row["status"] = "pending"
        elif statuses == {"complete"}:
            row["status"] = "complete"
        else:
            row["status"] = "partial"
    selected_methods = included_methods or tuple(METHODS)
    if any(method not in rows for method in selected_methods):
        raise SchemaError("included_methods contains an unknown method")
    notes = [
        "null means no value was present in a validated input artifact",
        "archived Project inference is node-to-node and is intentionally excluded",
        "timing boundaries are reported verbatim or composed only from explicit batch/repetition fields",
        "endpoint_mismatches is a compatibility alias of endpoint_failures",
        "artifact bytes and hashes are measured only for files explicitly listed in the input manifest or absolutely bound by validated diagnostics",
        "method status combines quality and efficiency status; artifact evidence has its own independent status for v1 compatibility",
    ]
    notes.append("method rows follow the registered seven-method publication set")
    return {
        "schema": OUTPUT_SCHEMA,
        "methods": [rows[method] for method in selected_methods],
        "notes": notes,
    }


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _phase_devices(efficiency: Mapping[str, Any]) -> str | None:
    values = []
    for phase, label in (("offline", "off"), ("training", "train"), ("prediction", "pred")):
        device = efficiency[f"{phase}_device"]
        if device is not None:
            values.append(f"{label}:{device}")
    return "; ".join(values) if values else None


def _artifact_hashes(artifacts: Mapping[str, Any], kind: str) -> str | None:
    values = [
        f"{item['role']}={item['sha256']}"
        for item in artifacts["items"]
        if item["kind"] == kind and item["sha256"] is not None
    ]
    return "; ".join(values) if values else None


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Route-baseline results",
        "",
        "Missing runs and unsupported cells are shown as —; no published or estimated values are substituted.",
        "",
        "## Route quality",
        "",
        "| Method | Status | N | P | R | F1 | Jaccard | Exact | Endpoint failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        quality = row["quality"]
        lines.append(
            "| " + " | ".join(
                (
                    row["label"], quality["status"],
                    _cell(quality["sample_count"], 0),
                    _cell(quality["edge_precision"]), _cell(quality["edge_recall"]),
                    _cell(quality["edge_f1"]), _cell(quality["edge_jaccard"]),
                    _cell(quality["exact_match"]),
                    _cell(quality["endpoint_failures"], 0),
                )
            ) + " |"
        )
    lines.extend(
        (
            "", "## Training and inference efficiency", "",
            "| Method | Status | Offline s | Train total s | Threads | Devices (off/train/pred) | Batch boundary | Batch s | ms/query | QPS | Peak RSS KiB (off/train/pred) | GPU peak bytes (train/pred) |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in summary["methods"]:
        efficiency = row["efficiency"]
        rss = "/".join(
            _cell(efficiency[key], 0)
            for key in ("offline_peak_rss_kib", "training_peak_rss_kib", "prediction_peak_rss_kib")
        )
        gpu = "/".join(
            _cell(efficiency[key], 0)
            for key in (
                "training_peak_gpu_memory_bytes",
                "prediction_peak_gpu_memory_bytes",
            )
        )
        lines.append(
            "| " + " | ".join(
                (
                    row["label"], efficiency["status"],
                    _cell(efficiency["offline_seconds"], 3),
                    _cell(efficiency["training_total_seconds"], 3),
                    _cell(efficiency["threads"], 0),
                    _cell(_phase_devices(efficiency)),
                    _cell(efficiency["batch_boundary"]),
                    _cell(efficiency["mean_batch_seconds"], 6),
                    _cell(efficiency["mean_ms_per_query"], 4),
                    _cell(efficiency["queries_per_second"], 2), rss, gpu,
                )
            ) + " |"
        )
    lines.extend(
        (
            "", "## Artifact evidence", "",
            "| Method | Status | Model bytes | Auxiliary bytes | Total bytes | Model SHA-256 identities | Auxiliary SHA-256 identities |",
            "|---|---:|---:|---:|---:|---|---|",
        )
    )
    for row in summary["methods"]:
        artifacts = row["artifacts"]
        lines.append(
            "| " + " | ".join(
                (
                    row["label"], artifacts["status"],
                    _cell(artifacts["model_bytes"], 0),
                    _cell(artifacts["auxiliary_bytes"], 0),
                    _cell(artifacts["model_and_auxiliary_bytes"], 0),
                    _cell(_artifact_hashes(artifacts, "model")),
                    _cell(_artifact_hashes(artifacts, "auxiliary")),
                )
            ) + " |"
        )
    lines.extend(
        (
            "", "Notes:", "",
            "- Offline and training totals retain their source boundaries; they are not silently added.",
            "- RSS is shown as offline/training/prediction peak KiB.",
            "- GPU memory is shown as training/prediction peak allocated bytes; — means unavailable or not applicable according to the method phase.",
            "- Archived Project edge-to-edge quality/training may be used, but its node-to-node inference timing is never promoted into this table.",
        )
    )
    lines.extend(
        (
            "- DRNCS-LG is the registered directed-line-graph edge-state adaptation of node-state DRNCS.",
            "- DRPK-static is the registered equal-information, time-collapsed adaptation; DRP-TP is its non-learned planning component and has no training phase.",
            "- Artifact totals include only files explicitly declared in the input manifest or absolutely bound by validated diagnostics; missing identities are never inferred from filenames.",
            "- The method status combines route quality and efficiency; artifact evidence is reported with its own status so v1 status semantics remain stable.",
            "",
        )
    )
    return "\n".join(lines)


def render_csv(summary: Mapping[str, Any]) -> str:
    """Render one flat, machine-friendly paper row per registered method."""

    columns = (
        "method_id", "method", "status", "quality_status", "sample_count",
        "edge_precision", "edge_recall", "edge_f1", "edge_jaccard",
        "exact_match", "endpoint_failures", "efficiency_status",
        "offline_seconds", "training_total_seconds", "threads",
        "offline_device", "training_device", "prediction_device",
        "batch_boundary", "mean_batch_seconds", "mean_ms_per_query",
        "queries_per_second", "offline_peak_rss_kib",
        "training_peak_rss_kib", "prediction_peak_rss_kib",
        "training_peak_gpu_memory_bytes", "prediction_peak_gpu_memory_bytes",
        "artifact_status", "model_bytes", "auxiliary_bytes",
        "model_and_auxiliary_bytes", "model_sha256_identities",
        "auxiliary_sha256_identities",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in summary["methods"]:
        quality, efficiency, artifacts = (
            row["quality"], row["efficiency"], row["artifacts"]
        )
        writer.writerow(
            {
                "method_id": row["id"], "method": row["label"],
                "status": row["status"], "quality_status": quality["status"],
                "sample_count": quality["sample_count"],
                "edge_precision": quality["edge_precision"],
                "edge_recall": quality["edge_recall"],
                "edge_f1": quality["edge_f1"],
                "edge_jaccard": quality["edge_jaccard"],
                "exact_match": quality["exact_match"],
                "endpoint_failures": quality["endpoint_failures"],
                "efficiency_status": efficiency["status"],
                "offline_seconds": efficiency["offline_seconds"],
                "training_total_seconds": efficiency["training_total_seconds"],
                "threads": efficiency["threads"],
                "offline_device": efficiency["offline_device"],
                "training_device": efficiency["training_device"],
                "prediction_device": efficiency["prediction_device"],
                "batch_boundary": efficiency["batch_boundary"],
                "mean_batch_seconds": efficiency["mean_batch_seconds"],
                "mean_ms_per_query": efficiency["mean_ms_per_query"],
                "queries_per_second": efficiency["queries_per_second"],
                "offline_peak_rss_kib": efficiency["offline_peak_rss_kib"],
                "training_peak_rss_kib": efficiency["training_peak_rss_kib"],
                "prediction_peak_rss_kib": efficiency["prediction_peak_rss_kib"],
                "training_peak_gpu_memory_bytes": efficiency["training_peak_gpu_memory_bytes"],
                "prediction_peak_gpu_memory_bytes": efficiency["prediction_peak_gpu_memory_bytes"],
                "artifact_status": artifacts["status"],
                "model_bytes": artifacts["model_bytes"],
                "auxiliary_bytes": artifacts["auxiliary_bytes"],
                "model_and_auxiliary_bytes": artifacts["model_and_auxiliary_bytes"],
                "model_sha256_identities": _artifact_hashes(artifacts, "model"),
                "auxiliary_sha256_identities": _artifact_hashes(artifacts, "auxiliary"),
            }
        )
    return output.getvalue()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Input manifest (method paths are relative to runtime_root;
archived_summary is relative to the manifest):
  {
    "schema": "ewr.route-baseline-summary-input/v1",
    "runtime_root": "../../generated/full_data_comparison_20260718",
    "archived_summary": "../archive-summary.json",
    "included_methods": ["project", "sp_length", "neuromlr_greedy"],
    "methods": {
      "project": {
        "evaluation": "project/test.evaluation.json",
        "training": "project/training-summary.json",
        "prediction": "project/test.diagnostics.json",
        "artifacts": [
          {"kind": "model", "role": "selected_checkpoint",
           "path": "project/checkpoint-selected.json"}
        ]
      },
      "drncs_lg": {"evaluation": null, "training": null}
    }
  }

Omitted methods and null artifact paths remain pending. Unknown methods,
unknown keys, duplicate JSON keys, unsupported schemas, and missing explicit
files are errors.
""",
    )
    parser.add_argument(
        "--input-manifest", type=Path,
        help="strict manifest for explicit Project/DRNCS/DRPK and overrides",
    )
    parser.add_argument(
        "--runtime-root", type=Path,
        help="runtime artifact root; overrides runtime_root in the manifest",
    )
    archive = parser.add_mutually_exclusive_group()
    archive.add_argument(
        "--archived-summary", type=Path,
        help="legacy Project/NeuroMLR-G summary (default: repository archive)",
    )
    archive.add_argument(
        "--no-archive", action="store_true",
        help="disable all legacy archive fallback",
    )
    parser.add_argument(
        "--summary-output", type=Path, required=True,
        help="machine-readable ewr.route-baseline-summary/v1 JSON",
    )
    parser.add_argument(
        "--markdown-output", type=Path, required=True,
        help="paper-style quality and efficiency Markdown tables",
    )
    parser.add_argument(
        "--csv-output", type=Path,
        help="optional flat paper-table CSV (one row per registered method)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.input_manifest.resolve() if args.input_manifest else None
        runtime_root, paths, manifest_archive, included_methods = _load_manifest(
            manifest_path, args.runtime_root
        )
        del runtime_root  # resolution is reflected in the source paths
        if args.no_archive:
            archive_path = None
        elif args.archived_summary is not None:
            archive_path = args.archived_summary.resolve()
            if not archive_path.is_file():
                raise SchemaError(f"archive file does not exist: {archive_path}")
        elif isinstance(manifest_archive, Path):
            archive_path = manifest_archive
        elif manifest_archive is None:
            archive_path = None
        else:
            archive_path = DEFAULT_ARCHIVE if DEFAULT_ARCHIVE.is_file() else None
        summary = summarize(paths, archive_path, included_methods)
        _write_text_atomic(
            args.summary_output,
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        _write_text_atomic(args.markdown_output, render_markdown(summary))
        if args.csv_output is not None:
            _write_text_atomic(args.csv_output, render_csv(summary))
    except SchemaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
