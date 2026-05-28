from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - import guard for environments without h5py
    h5py = None

try:
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - gzip-only files can still work without it
    hdf5plugin = None

import numpy as np


COMMON_REQUIRED_DATASETS = (
    "voxels",
    "window_t_start_us",
    "window_t_end_us",
    "window_event_count",
    "anchor_timestamp_us",
)
COMMON_OPTIONAL_LENGTH_MATCH_DATASETS = (
    "window_index",
    "window_rel_start_us",
    "window_rel_end_us",
    "anchor_rel_timestamp_us",
    "window_activity_score",
    "window_active_pixel_ratio",
    "window_activity_grid",
    "embedded_segmentation",
    "embedded_semantics",
    "embedded_depth",
    "segmentation_available",
    "segmentation_timestamp_us",
    "segmentation_time_delta_us",
    "segmentation_relpath",
)
GENERIC_SEQUENCE_NAMES = {
    "",
    "train",
    "test",
    "val",
    "events",
    "event",
    "left",
    "right",
    "data",
    "voxel",
    "voxels",
    "images",
}


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to validate H5 files. Install project dependencies first.")


def _safe_attr(attrs: Any, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_voxel_h5(path: Path) -> bool:
    _require_h5py()
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4
    except Exception:
        return False


def _select_files(input_path: Path | None, dataset_root: Path | None, recursive: bool) -> list[Path]:
    if input_path is not None and dataset_root is not None:
        raise ValueError("use either --input_path or --dataset_root, not both")
    if input_path is None and dataset_root is None:
        raise ValueError("either --input_path or --dataset_root is required")
    if input_path is not None:
        return [input_path]
    assert dataset_root is not None
    pattern = "**/*.h5" if recursive else "*.h5"
    return sorted([path for path in dataset_root.glob(pattern) if path.is_file()])


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _strip_known_suffixes(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"_part\d+$", "", text)
    changed = True
    while changed:
        changed = False
        for suffix in (
            "_left_event",
            "_right_event",
            "_events",
            "_event",
            "_voxels_semantic",
            "_voxels_depth",
            "_voxels",
            "_1x",
            "_2x",
        ):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def _is_generic_sequence_name(name: str) -> bool:
    return str(name).strip().lower() in GENERIC_SEQUENCE_NAMES


def _guess_sequence_name(*, file_path: Path, relative_file: str) -> str:
    rel_path = Path(relative_file)
    stem = _strip_known_suffixes(rel_path.stem)
    if len(stem) > 0 and not _is_generic_sequence_name(stem):
        return stem
    for parent in rel_path.parents:
        name = _strip_known_suffixes(parent.name)
        if len(name) > 0 and not _is_generic_sequence_name(name):
            return name

    return _strip_known_suffixes(file_path.stem)


def _infer_dataset_family(file_path: Path, representation: str) -> str:
    rep = str(representation).strip().lower()
    if rep in {"event_voxel_grid_m3ed", "event_image_m3ed"}:
        return "m3ed"
    if rep in {"event_voxel_grid_1mpx", "event_image_1mpx"}:
        return "1mpx"
    if rep in {"event_voxel_grid_eventscape", "event_image_eventscape"}:
        return "eventscape"
    if rep in {"event_voxel_grid", "event_image"}:
        return "dsec"

    lower_path = str(file_path).lower()
    if "m3ed" in lower_path:
        return "m3ed"
    if "1mpx" in lower_path:
        return "1mpx"
    if "eventscape" in lower_path:
        return "eventscape"
    if "dsec" in lower_path:
        return "dsec"
    return "other"


def _pick_sample_indices(n_samples: int, count: int) -> np.ndarray:
    if n_samples <= 0 or count <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_samples <= count:
        return np.arange(n_samples, dtype=np.int64)
    return np.unique(np.linspace(0, n_samples - 1, num=count, dtype=np.int64))


def _read_1d_dataset(ds: Any) -> np.ndarray:
    return np.asarray(ds[()], dtype=np.float64).reshape(-1)


def _summary_value(values: list[float]) -> dict[str, float | None]:
    if len(values) == 0:
        return {"min": None, "p5": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(arr.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    file_path: Path,
    relative_file: str,
    dataset_family: str,
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append(
        {
            "file": str(file_path),
            "relative_file": relative_file,
            "dataset_family": dataset_family,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def _check_monotonic_increasing(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.all(values[1:] >= values[:-1]))


def _check_ratio_range(name: str, values: np.ndarray, issues: list[dict[str, Any]], *, file_path: Path, relative_file: str, dataset_family: str) -> None:
    if values.size == 0:
        return
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        _add_issue(
            issues,
            file_path=file_path,
            relative_file=relative_file,
            dataset_family=dataset_family,
            severity="error",
            code=f"{name}_nonfinite",
            message=f"{name} contains NaN/Inf values.",
        )
    if finite.size > 0 and (float(finite.min()) < -1e-6 or float(finite.max()) > 1.0 + 1e-6):
        _add_issue(
            issues,
            file_path=file_path,
            relative_file=relative_file,
            dataset_family=dataset_family,
            severity="error",
            code=f"{name}_out_of_range",
            message=f"{name} is expected to stay within [0, 1], but saw min={float(finite.min()):.6f}, max={float(finite.max()):.6f}.",
        )


def _validate_file(
    file_path: Path,
    *,
    dataset_root: Path | None,
    sample_windows_per_file: int,
    warn_zero_event_ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_h5py()
    relative_file = _relative_or_name(file_path, dataset_root)
    issues: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "file": str(file_path),
        "relative_file": relative_file,
        "dataset_family": "other",
        "representation": "",
        "sync_target": "",
        "samples": 0,
        "voxel_channels": None,
        "height": None,
        "width": None,
        "mean_event_count": None,
        "zero_event_ratio": None,
        "mean_active_pixel_ratio": None,
        "mean_activity_score": None,
        "sample_nonfinite_voxel_count": None,
        "sample_all_zero_voxel_ratio": None,
        "resolved_semantics_ts_source": "",
        "resolved_depth_ts_source": "",
        "embedded_label_dataset": "",
        "embedded_label_source_path": "",
        "sequence_name": "",
        "low_activity_suspicious": 0,
        "low_activity_reasons": "",
        "status": "ok",
        "num_errors": 0,
        "num_warnings": 0,
    }

    try:
        with h5py.File(str(file_path), "r") as h5f:
            representation = str(_safe_attr(h5f.attrs, "representation", ""))
            dataset_family = _infer_dataset_family(file_path, representation)
            sync_target = str(_safe_attr(h5f.attrs, "sync_target", ""))
            row["dataset_family"] = dataset_family
            row["representation"] = representation
            row["sync_target"] = sync_target
            row["resolved_semantics_ts_source"] = str(_safe_attr(h5f.attrs, "resolved_semantics_ts_source", ""))
            row["resolved_depth_ts_source"] = str(_safe_attr(h5f.attrs, "resolved_depth_ts_source", ""))
            row["embedded_label_dataset"] = str(_safe_attr(h5f.attrs, "embedded_label_dataset", ""))
            row["embedded_label_source_path"] = str(_safe_attr(h5f.attrs, "embedded_label_source_path", ""))
            row["sequence_name"] = _guess_sequence_name(
                file_path=file_path,
                relative_file=relative_file,
            )

            if "voxels" not in h5f:
                _add_issue(
                    issues,
                    file_path=file_path,
                    relative_file=relative_file,
                    dataset_family=dataset_family,
                    severity="error",
                    code="missing_voxels",
                    message="Missing required dataset 'voxels'.",
                )
                return row, issues

            voxels = h5f["voxels"]
            if voxels.ndim != 4:
                _add_issue(
                    issues,
                    file_path=file_path,
                    relative_file=relative_file,
                    dataset_family=dataset_family,
                    severity="error",
                    code="invalid_voxel_rank",
                    message=f"'voxels' must be rank-4 (N,C,H,W), got shape={tuple(voxels.shape)}.",
                )
                return row, issues

            samples = int(voxels.shape[0])
            row["samples"] = samples
            row["voxel_channels"] = int(voxels.shape[1])
            row["height"] = int(voxels.shape[2])
            row["width"] = int(voxels.shape[3])

            if samples <= 0:
                _add_issue(
                    issues,
                    file_path=file_path,
                    relative_file=relative_file,
                    dataset_family=dataset_family,
                    severity="error",
                    code="empty_h5",
                    message="No windows were written to this H5.",
                )

            for name in COMMON_REQUIRED_DATASETS:
                if name not in h5f:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code=f"missing_{name}",
                        message=f"Missing required dataset '{name}'.",
                    )
                elif int(h5f[name].shape[0]) != samples:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code=f"length_mismatch_{name}",
                        message=f"Dataset '{name}' has length {int(h5f[name].shape[0])}, expected {samples}.",
                    )

            for name in COMMON_OPTIONAL_LENGTH_MATCH_DATASETS:
                if name in h5f and int(h5f[name].shape[0]) != samples:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code=f"length_mismatch_{name}",
                        message=f"Dataset '{name}' has length {int(h5f[name].shape[0])}, expected {samples}.",
                    )

            if "window_t_start_us" in h5f and "window_t_end_us" in h5f:
                starts = _read_1d_dataset(h5f["window_t_start_us"])
                ends = _read_1d_dataset(h5f["window_t_end_us"])
                if starts.size == samples and ends.size == samples:
                    if not _check_monotonic_increasing(starts):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="window_start_not_monotonic",
                            message="window_t_start_us is not monotonic non-decreasing.",
                        )
                    if not _check_monotonic_increasing(ends):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="window_end_not_monotonic",
                            message="window_t_end_us is not monotonic non-decreasing.",
                        )
                    if np.any(ends <= starts):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="window_nonpositive_duration",
                            message="Found windows where end <= start.",
                        )

            if "window_index" in h5f:
                window_index = _read_1d_dataset(h5f["window_index"])
                if window_index.size == samples and not _check_monotonic_increasing(window_index):
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="warning",
                        code="window_index_not_monotonic",
                        message="window_index is not monotonic non-decreasing.",
                    )

            if "window_event_count" in h5f:
                event_counts = _read_1d_dataset(h5f["window_event_count"])
                if event_counts.size == samples:
                    if np.any(event_counts < 0):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="negative_event_count",
                            message="window_event_count contains negative values.",
                        )
                    row["mean_event_count"] = float(event_counts.mean()) if event_counts.size > 0 else None
                    zero_event_ratio = float(np.mean(event_counts <= 0)) if event_counts.size > 0 else None
                    row["zero_event_ratio"] = zero_event_ratio
                    if event_counts.size > 0 and np.all(event_counts <= 0):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="all_zero_event_windows",
                            message="All windows have zero event_count.",
                        )
                    elif zero_event_ratio is not None and zero_event_ratio >= float(warn_zero_event_ratio):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="warning",
                            code="high_zero_event_ratio",
                            message=f"zero_event_ratio={zero_event_ratio:.4f} exceeded warn threshold {float(warn_zero_event_ratio):.4f}.",
                        )

            has_activity = False
            if "window_active_pixel_ratio" in h5f:
                active_ratio = _read_1d_dataset(h5f["window_active_pixel_ratio"])
                if active_ratio.size == samples:
                    has_activity = True
                    _check_ratio_range(
                        "window_active_pixel_ratio",
                        active_ratio,
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                    )
                    finite = active_ratio[np.isfinite(active_ratio)]
                    row["mean_active_pixel_ratio"] = float(finite.mean()) if finite.size > 0 else None
                    if finite.size > 0 and np.all(finite == 0):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="warning",
                            code="all_zero_active_pixel_ratio",
                            message="window_active_pixel_ratio is zero for all windows.",
                        )
            if "window_activity_score" in h5f:
                activity_score = _read_1d_dataset(h5f["window_activity_score"])
                if activity_score.size == samples:
                    has_activity = True
                    _check_ratio_range(
                        "window_activity_score",
                        activity_score,
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                    )
                    finite = activity_score[np.isfinite(activity_score)]
                    row["mean_activity_score"] = float(finite.mean()) if finite.size > 0 else None
                    if finite.size > 0 and np.all(finite == 0):
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="warning",
                            code="all_zero_activity_score",
                            message="window_activity_score is zero for all windows.",
                        )
            if not has_activity:
                _add_issue(
                    issues,
                    file_path=file_path,
                    relative_file=relative_file,
                    dataset_family=dataset_family,
                    severity="warning",
                    code="missing_activity_metadata",
                    message="window_activity_score / window_active_pixel_ratio were not found.",
                )

            sample_indices = _pick_sample_indices(samples, sample_windows_per_file)
            if sample_indices.size > 0:
                sample_voxels = np.asarray(voxels[sample_indices], dtype=np.float32)
                nonfinite_count = int(np.count_nonzero(~np.isfinite(sample_voxels)))
                row["sample_nonfinite_voxel_count"] = nonfinite_count
                if nonfinite_count > 0:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code="nonfinite_voxels",
                        message=f"Sampled voxels contained {nonfinite_count} NaN/Inf values.",
                    )
                flattened = sample_voxels.reshape(sample_voxels.shape[0], -1)
                all_zero_ratio = float(np.mean(np.all(flattened == 0, axis=1)))
                row["sample_all_zero_voxel_ratio"] = all_zero_ratio
                if sample_voxels.shape[0] > 0 and np.all(np.all(flattened == 0, axis=1)):
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="warning",
                        code="sampled_voxels_all_zero",
                        message="All sampled voxel windows were entirely zero.",
                    )

            if sync_target in {"semantic", "depth"}:
                embedded_label_dataset = str(_safe_attr(h5f.attrs, "embedded_label_dataset", ""))
                if len(embedded_label_dataset) == 0:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code="missing_label_reference",
                        message=f"sync_target={sync_target} but no embedded label dataset was recorded.",
                    )
                if len(embedded_label_dataset) > 0 and embedded_label_dataset not in h5f:
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code="missing_embedded_label_dataset",
                        message=f"embedded label dataset '{embedded_label_dataset}' was declared but is missing.",
                    )
                if len(embedded_label_dataset) > 0 and embedded_label_dataset in h5f:
                    if int(h5f[embedded_label_dataset].shape[0]) != samples:
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="embedded_label_length_mismatch",
                            message=(
                                f"embedded label dataset '{embedded_label_dataset}' has length "
                                f"{int(h5f[embedded_label_dataset].shape[0])}, expected {samples}."
                            ),
                        )

            if dataset_family == "m3ed" and sync_target == "semantic":
                resolved_source = str(_safe_attr(h5f.attrs, "resolved_semantics_ts_source", ""))
                num_semantic_timestamps = int(_safe_attr(h5f.attrs, "num_semantic_timestamps", 0) or 0)
                if resolved_source == "ovc_ts_map":
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="warning",
                        code="m3ed_semantic_ovc_fallback",
                        message="M3ED semantic preprocessing resolved to ovc_ts_map; this is a known high-risk fallback.",
                    )
                if samples > 0 and num_semantic_timestamps not in (0, samples):
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code="m3ed_semantic_timestamp_count_mismatch",
                        message=f"num_semantic_timestamps={num_semantic_timestamps}, expected {samples}.",
                    )

            if dataset_family == "m3ed" and sync_target == "depth":
                num_depth_timestamps = int(_safe_attr(h5f.attrs, "num_depth_timestamps", 0) or 0)
                if samples > 0 and num_depth_timestamps not in (0, samples):
                    _add_issue(
                        issues,
                        file_path=file_path,
                        relative_file=relative_file,
                        dataset_family=dataset_family,
                        severity="error",
                        code="m3ed_depth_timestamp_count_mismatch",
                        message=f"num_depth_timestamps={num_depth_timestamps}, expected {samples}.",
                    )

            if dataset_family == "dsec" and "segmentation_available" in h5f:
                segmentation_available = _read_1d_dataset(h5f["segmentation_available"])
                if segmentation_available.size == samples and sync_target != "event_only":
                    keep_ratio = float(np.mean(segmentation_available > 0))
                    row["segmentation_available_ratio"] = keep_ratio
                    embedded_label_dataset = str(_safe_attr(h5f.attrs, "embedded_label_dataset", ""))
                    if embedded_label_dataset != "embedded_segmentation" or "embedded_segmentation" not in h5f:
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="error",
                            code="dsec_missing_embedded_segmentation",
                            message="DSEC semantic preprocessing must embed labels into 'embedded_segmentation'.",
                        )
                    if keep_ratio == 0.0:
                        _add_issue(
                            issues,
                            file_path=file_path,
                            relative_file=relative_file,
                            dataset_family=dataset_family,
                            severity="warning",
                            code="dsec_no_segmentation_matches",
                            message="segmentation_available is zero for all windows.",
                        )

            if dataset_family == "eventscape":
                if "semantic_available" in h5f:
                    semantic_available = _read_1d_dataset(h5f["semantic_available"])
                    if semantic_available.size == samples and np.any(semantic_available > 0):
                        embedded_semantic_dataset = str(_safe_attr(h5f.attrs, "embedded_semantic_dataset", ""))
                        if embedded_semantic_dataset != "embedded_semantics" or "embedded_semantics" not in h5f:
                            _add_issue(
                                issues,
                                file_path=file_path,
                                relative_file=relative_file,
                                dataset_family=dataset_family,
                                severity="error",
                                code="eventscape_missing_embedded_semantics",
                                message="EventScape semantic labels must be embedded into 'embedded_semantics'.",
                            )
                        elif int(h5f["embedded_semantics"].shape[0]) != samples:
                            _add_issue(
                                issues,
                                file_path=file_path,
                                relative_file=relative_file,
                                dataset_family=dataset_family,
                                severity="error",
                                code="eventscape_embedded_semantics_length_mismatch",
                                message=(
                                    f"embedded_semantics has length {int(h5f['embedded_semantics'].shape[0])}, "
                                    f"expected {samples}."
                                ),
                            )
                if "depth_available" in h5f:
                    depth_available = _read_1d_dataset(h5f["depth_available"])
                    if depth_available.size == samples and np.any(depth_available > 0):
                        embedded_depth_dataset = str(_safe_attr(h5f.attrs, "embedded_depth_dataset", ""))
                        if embedded_depth_dataset != "embedded_depth" or "embedded_depth" not in h5f:
                            _add_issue(
                                issues,
                                file_path=file_path,
                                relative_file=relative_file,
                                dataset_family=dataset_family,
                                severity="error",
                                code="eventscape_missing_embedded_depth",
                                message="EventScape depth labels must be embedded into 'embedded_depth'.",
                            )
                        elif int(h5f["embedded_depth"].shape[0]) != samples:
                            _add_issue(
                                issues,
                                file_path=file_path,
                                relative_file=relative_file,
                                dataset_family=dataset_family,
                                severity="error",
                                code="eventscape_embedded_depth_length_mismatch",
                                message=(
                                    f"embedded_depth has length {int(h5f['embedded_depth'].shape[0])}, "
                                    f"expected {samples}."
                                ),
                            )
    except Exception as exc:
        _add_issue(
            issues,
            file_path=file_path,
            relative_file=relative_file,
            dataset_family=row["dataset_family"],
            severity="error",
            code="exception",
            message=str(exc),
        )

    num_errors = sum(1 for issue in issues if issue["severity"] == "error")
    num_warnings = sum(1 for issue in issues if issue["severity"] == "warning")
    row["num_errors"] = num_errors
    row["num_warnings"] = num_warnings
    row["status"] = "error" if num_errors > 0 else "warning" if num_warnings > 0 else "ok"
    return row, issues


def _top_rows(rows: list[dict[str, Any]], key: str, *, reverse: bool, limit: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get(key) is not None]
    filtered.sort(key=lambda row: float(row[key]), reverse=reverse)
    return filtered[:limit]


def _safe_percentile(values: list[float], q: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _annotate_low_activity_outliers(
    *,
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    low_activity_active_percentile: float,
    low_activity_event_percentile: float,
    low_activity_zero_event_percentile: float,
    low_activity_min_files_for_cluster: int,
    low_activity_warn_suspicious_file_ratio: float | None,
    low_activity_warn_top_sequence_share: float | None,
    low_activity_error_suspicious_file_ratio: float | None,
    low_activity_error_top_sequence_share: float | None,
) -> dict[str, Any]:
    active_values = [float(row["mean_active_pixel_ratio"]) for row in rows if row.get("mean_active_pixel_ratio") is not None]
    event_values = [float(row["mean_event_count"]) for row in rows if row.get("mean_event_count") is not None]
    zero_values = [float(row["zero_event_ratio"]) for row in rows if row.get("zero_event_ratio") is not None]

    active_threshold = _safe_percentile(active_values, float(low_activity_active_percentile))
    event_threshold = _safe_percentile(event_values, float(low_activity_event_percentile))
    zero_threshold = _safe_percentile(zero_values, float(low_activity_zero_event_percentile))

    suspicious_rows: list[dict[str, Any]] = []
    sequence_counts: dict[str, int] = {}
    for row in rows:
        reasons: list[str] = []
        mean_active = row.get("mean_active_pixel_ratio")
        mean_events = row.get("mean_event_count")
        zero_ratio = row.get("zero_event_ratio")

        if mean_active is not None and active_threshold is not None and float(mean_active) <= float(active_threshold):
            reasons.append(f"mean_active_pixel_ratio<={float(active_threshold):.6f}")
        if mean_events is not None and event_threshold is not None and float(mean_events) <= float(event_threshold):
            reasons.append(f"mean_event_count<={float(event_threshold):.3f}")
        if zero_ratio is not None and zero_threshold is not None and float(zero_ratio) >= float(zero_threshold):
            reasons.append(f"zero_event_ratio>={float(zero_threshold):.6f}")

        suspicious = False
        if zero_ratio is not None and zero_threshold is not None and float(zero_ratio) >= float(zero_threshold):
            suspicious = True
        elif len(reasons) >= 2:
            suspicious = True

        row["low_activity_suspicious"] = int(suspicious)
        row["low_activity_reasons"] = ",".join(reasons)
        if suspicious:
            suspicious_rows.append(row)
            sequence_name = str(row.get("sequence_name", ""))
            sequence_counts[sequence_name] = sequence_counts.get(sequence_name, 0) + 1
            row["num_warnings"] = int(row.get("num_warnings", 0)) + 1
            if str(row.get("status", "ok")) == "ok":
                row["status"] = "warning"
            _add_issue(
                issues,
                file_path=Path(str(row["file"])),
                relative_file=str(row["relative_file"]),
                dataset_family=str(row["dataset_family"]),
                severity="warning",
                code="low_activity_outlier",
                message=(
                    "This file looks like a low-activity outlier. "
                    f"reasons={','.join(reasons) if len(reasons) > 0 else 'n/a'}"
                ),
            )

    suspicious_count = len(suspicious_rows)
    suspicious_ratio = float(suspicious_count) / float(len(rows)) if len(rows) > 0 else 0.0
    top_sequence = ""
    top_sequence_count = 0
    for sequence_name, count in sequence_counts.items():
        if count > top_sequence_count:
            top_sequence = sequence_name
            top_sequence_count = count
    top_sequence_share = float(top_sequence_count) / float(suspicious_count) if suspicious_count > 0 else 0.0

    cluster_message = None
    cluster_severity = None
    if suspicious_count >= int(low_activity_min_files_for_cluster):
        warn_hit = (
            low_activity_warn_suspicious_file_ratio is not None
            and low_activity_warn_top_sequence_share is not None
            and suspicious_ratio >= float(low_activity_warn_suspicious_file_ratio)
            and top_sequence_share >= float(low_activity_warn_top_sequence_share)
        )
        error_hit = (
            low_activity_error_suspicious_file_ratio is not None
            and low_activity_error_top_sequence_share is not None
            and suspicious_ratio >= float(low_activity_error_suspicious_file_ratio)
            and top_sequence_share >= float(low_activity_error_top_sequence_share)
        )
        if error_hit or warn_hit:
            cluster_severity = "error" if error_hit else "warning"
            cluster_message = (
                "Low-activity outliers are concentrated in a small subset of sequences. "
                f"suspicious_files={suspicious_count}/{len(rows)} ({suspicious_ratio:.4f}), "
                f"top_sequence={top_sequence or 'n/a'} ({top_sequence_count}/{suspicious_count}, {top_sequence_share:.4f})."
            )
            _add_issue(
                issues,
                file_path=Path("__dataset__"),
                relative_file="__dataset__",
                dataset_family="mixed",
                severity=cluster_severity,
                code="low_activity_clustered",
                message=cluster_message,
            )

    return {
        "active_threshold": active_threshold,
        "event_threshold": event_threshold,
        "zero_event_threshold": zero_threshold,
        "suspicious_file_count": suspicious_count,
        "suspicious_file_ratio": suspicious_ratio,
        "top_sequence": top_sequence,
        "top_sequence_count": top_sequence_count,
        "top_sequence_share": top_sequence_share,
        "cluster_issue_severity": cluster_severity,
        "cluster_issue_message": cluster_message,
    }


def _write_markdown_report(
    output_path: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    top_zero = _top_rows(rows, "zero_event_ratio", reverse=True, limit=10)
    top_low_active = _top_rows(rows, "mean_active_pixel_ratio", reverse=False, limit=10)
    top_low_events = _top_rows(rows, "mean_event_count", reverse=False, limit=10)

    lines: list[str] = []
    lines.append("# Preprocess Healthcheck")
    lines.append("")
    lines.append(f"- files_scanned: {summary['num_files']}")
    lines.append(f"- ok_files: {summary['num_ok_files']}")
    lines.append(f"- warning_files: {summary['num_warning_files']}")
    lines.append(f"- error_files: {summary['num_error_files']}")
    lines.append(f"- warning_count: {summary['num_warnings']}")
    lines.append(f"- error_count: {summary['num_errors']}")
    lines.append("")
    lines.append("## Dataset Families")
    lines.append("")
    for key, value in sorted(summary["dataset_family_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Frequent Issues")
    lines.append("")
    if len(summary["issue_code_counts"]) == 0:
        lines.append("- none")
    else:
        for code, count in sorted(summary["issue_code_counts"].items(), key=lambda item: (-item[1], item[0]))[:20]:
            lines.append(f"- {code}: {count}")
    lines.append("")
    lines.append("## Low-Activity Outliers")
    lines.append("")
    low_activity = summary.get("low_activity_outlier_summary", {})
    lines.append(
        "- thresholds: "
        f"mean_active_pixel_ratio<={low_activity.get('active_threshold')} | "
        f"mean_event_count<={low_activity.get('event_threshold')} | "
        f"zero_event_ratio>={low_activity.get('zero_event_threshold')}"
    )
    lines.append(
        "- suspicious_files: "
        f"{low_activity.get('suspicious_file_count', 0)}/{summary['num_files']} "
        f"({float(low_activity.get('suspicious_file_ratio', 0.0)):.4f})"
    )
    lines.append(
        "- top_sequence: "
        f"{low_activity.get('top_sequence', '') or 'n/a'} "
        f"({low_activity.get('top_sequence_count', 0)} files, "
        f"share={float(low_activity.get('top_sequence_share', 0.0)):.4f})"
    )
    if low_activity.get("cluster_issue_message"):
        lines.append(
            f"- cluster_judgement: [{low_activity.get('cluster_issue_severity')}] "
            f"{low_activity.get('cluster_issue_message')}"
        )
    suspicious_rows = [row for row in rows if int(row.get("low_activity_suspicious", 0)) > 0]
    if len(suspicious_rows) == 0:
        lines.append("- suspicious examples: none")
    else:
        for row in suspicious_rows[:20]:
            lines.append(
                f"- {row['relative_file']}: sequence={row.get('sequence_name', '')}, reasons={row.get('low_activity_reasons', '')}"
            )
    lines.append("")
    lines.append("## Highest Zero-Event Ratios")
    lines.append("")
    if len(top_zero) == 0:
        lines.append("- none")
    else:
        for row in top_zero:
            lines.append(f"- {row['relative_file']}: zero_event_ratio={float(row['zero_event_ratio']):.4f}")
    lines.append("")
    lines.append("## Lowest Mean Active Pixel Ratios")
    lines.append("")
    if len(top_low_active) == 0:
        lines.append("- none")
    else:
        for row in top_low_active:
            lines.append(f"- {row['relative_file']}: mean_active_pixel_ratio={float(row['mean_active_pixel_ratio']):.6f}")
    lines.append("")
    lines.append("## Lowest Mean Event Counts")
    lines.append("")
    if len(top_low_events) == 0:
        lines.append("- none")
    else:
        for row in top_low_events:
            lines.append(f"- {row['relative_file']}: mean_event_count={float(row['mean_event_count']):.2f}")
    lines.append("")
    lines.append("## Files With Errors")
    lines.append("")
    error_rows = [row for row in rows if row["status"] == "error"]
    if len(error_rows) == 0:
        lines.append("- none")
    else:
        for row in error_rows[:50]:
            lines.append(f"- {row['relative_file']}: errors={row['num_errors']}, warnings={row['num_warnings']}")
    lines.append("")
    lines.append("## Files With Warnings")
    lines.append("")
    warning_rows = [row for row in rows if row["status"] == "warning"]
    if len(warning_rows) == 0:
        lines.append("- none")
    else:
        for row in warning_rows[:50]:
            lines.append(f"- {row['relative_file']}: errors={row['num_errors']}, warnings={row['num_warnings']}")
    lines.append("")
    lines.append("## First 50 Issues")
    lines.append("")
    if len(issues) == 0:
        lines.append("- none")
    else:
        for issue in issues[:50]:
            lines.append(
                f"- [{issue['severity']}] {issue['relative_file']} | {issue['code']} | {issue['message']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a structural healthcheck over preprocessed voxel H5 files.")
    parser.add_argument("--input_path", type=Path, default=None, help="Validate a single voxel H5.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Validate all voxel H5 files under a root.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tmp/preprocess_healthcheck"),
        help="Directory for CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into subdirectories when using --dataset_root.",
    )
    parser.add_argument(
        "--sample_windows_per_file",
        type=int,
        default=8,
        help="How many windows to sample per file when checking for NaN/Inf and all-zero voxels.",
    )
    parser.add_argument(
        "--warn_zero_event_ratio",
        type=float,
        default=0.5,
        help="Warn when the fraction of zero-event windows reaches this value.",
    )
    parser.add_argument(
        "--fail_on",
        choices=["error", "warning", "never"],
        default="error",
        help="Exit non-zero when errors exist, warnings exist, or never fail.",
    )
    parser.add_argument(
        "--low_activity_active_percentile",
        type=float,
        default=5.0,
        help="Percentile used to derive the low mean_active_pixel_ratio threshold across files.",
    )
    parser.add_argument(
        "--low_activity_event_percentile",
        type=float,
        default=5.0,
        help="Percentile used to derive the low mean_event_count threshold across files.",
    )
    parser.add_argument(
        "--low_activity_zero_event_percentile",
        type=float,
        default=95.0,
        help="Percentile used to derive the high zero_event_ratio threshold across files.",
    )
    parser.add_argument(
        "--low_activity_min_files_for_cluster",
        type=int,
        default=3,
        help="Minimum suspicious files before clustered low-activity warnings/errors are considered.",
    )
    parser.add_argument(
        "--low_activity_warn_suspicious_file_ratio",
        type=float,
        default=0.2,
        help="Warn if suspicious files exceed this ratio and are concentrated in one sequence.",
    )
    parser.add_argument(
        "--low_activity_warn_top_sequence_share",
        type=float,
        default=0.5,
        help="Warn if this share of suspicious files comes from the same sequence.",
    )
    parser.add_argument(
        "--low_activity_error_suspicious_file_ratio",
        type=float,
        default=None,
        help="Optional: escalate clustered low-activity outliers to an error above this suspicious-file ratio.",
    )
    parser.add_argument(
        "--low_activity_error_top_sequence_share",
        type=float,
        default=None,
        help="Optional: escalate clustered low-activity outliers to an error above this top-sequence share.",
    )
    args = parser.parse_args()

    input_path = None if args.input_path is None else Path(args.input_path).expanduser().resolve()
    dataset_root = None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    files = _select_files(input_path=input_path, dataset_root=dataset_root, recursive=bool(args.recursive))
    files = [path for path in files if _is_voxel_h5(path)]
    if len(files) == 0:
        raise FileNotFoundError("No valid preprocessed voxel H5 files were found.")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for file_path in files:
        row, file_issues = _validate_file(
            file_path=file_path,
            dataset_root=dataset_root,
            sample_windows_per_file=max(0, int(args.sample_windows_per_file)),
            warn_zero_event_ratio=float(args.warn_zero_event_ratio),
        )
        rows.append(row)
        issues.extend(file_issues)

    low_activity_outlier_summary = _annotate_low_activity_outliers(
        rows=rows,
        issues=issues,
        low_activity_active_percentile=float(args.low_activity_active_percentile),
        low_activity_event_percentile=float(args.low_activity_event_percentile),
        low_activity_zero_event_percentile=float(args.low_activity_zero_event_percentile),
        low_activity_min_files_for_cluster=int(args.low_activity_min_files_for_cluster),
        low_activity_warn_suspicious_file_ratio=None
        if args.low_activity_warn_suspicious_file_ratio is None
        else float(args.low_activity_warn_suspicious_file_ratio),
        low_activity_warn_top_sequence_share=None
        if args.low_activity_warn_top_sequence_share is None
        else float(args.low_activity_warn_top_sequence_share),
        low_activity_error_suspicious_file_ratio=None
        if args.low_activity_error_suspicious_file_ratio is None
        else float(args.low_activity_error_suspicious_file_ratio),
        low_activity_error_top_sequence_share=None
        if args.low_activity_error_top_sequence_share is None
        else float(args.low_activity_error_top_sequence_share),
    )

    num_errors = sum(1 for issue in issues if issue["severity"] == "error")
    num_warnings = sum(1 for issue in issues if issue["severity"] == "warning")
    dataset_family_counts: dict[str, int] = {}
    issue_code_counts: dict[str, int] = {}
    for row in rows:
        dataset_family_counts[str(row["dataset_family"])] = dataset_family_counts.get(str(row["dataset_family"]), 0) + 1
    for issue in issues:
        code = str(issue["code"])
        issue_code_counts[code] = issue_code_counts.get(code, 0) + 1

    summary = {
        "num_files": len(rows),
        "num_ok_files": sum(1 for row in rows if row["status"] == "ok"),
        "num_warning_files": sum(1 for row in rows if row["status"] == "warning"),
        "num_error_files": sum(1 for row in rows if row["status"] == "error"),
        "num_warnings": num_warnings,
        "num_errors": num_errors,
        "dataset_family_counts": dataset_family_counts,
        "issue_code_counts": issue_code_counts,
        "mean_event_count_summary": _summary_value(
            [float(row["mean_event_count"]) for row in rows if row.get("mean_event_count") is not None]
        ),
        "zero_event_ratio_summary": _summary_value(
            [float(row["zero_event_ratio"]) for row in rows if row.get("zero_event_ratio") is not None]
        ),
        "mean_active_pixel_ratio_summary": _summary_value(
            [float(row["mean_active_pixel_ratio"]) for row in rows if row.get("mean_active_pixel_ratio") is not None]
        ),
        "mean_activity_score_summary": _summary_value(
            [float(row["mean_activity_score"]) for row in rows if row.get("mean_activity_score") is not None]
        ),
        "low_activity_outlier_summary": low_activity_outlier_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    row_columns = [
        "file",
        "relative_file",
        "dataset_family",
        "representation",
        "sync_target",
        "samples",
        "voxel_channels",
        "height",
        "width",
        "mean_event_count",
        "zero_event_ratio",
        "mean_active_pixel_ratio",
        "mean_activity_score",
        "sample_nonfinite_voxel_count",
        "sample_all_zero_voxel_ratio",
        "resolved_semantics_ts_source",
        "resolved_depth_ts_source",
        "embedded_label_dataset",
        "embedded_label_source_path",
        "sequence_name",
        "low_activity_suspicious",
        "low_activity_reasons",
        "segmentation_available_ratio",
        "num_errors",
        "num_warnings",
        "status",
    ]
    issue_columns = ["file", "relative_file", "dataset_family", "severity", "code", "message"]

    _write_csv(output_dir / "preprocess_health_rows.csv", rows, row_columns)
    _write_csv(output_dir / "preprocess_health_issues.csv", issues, issue_columns)
    (output_dir / "preprocess_health_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_markdown_report(output_dir / "preprocess_health_report.md", summary=summary, rows=rows, issues=issues)

    print(
        "[HEALTHCHECK] "
        f"files={summary['num_files']} ok={summary['num_ok_files']} "
        f"warning_files={summary['num_warning_files']} error_files={summary['num_error_files']} "
        f"warnings={summary['num_warnings']} errors={summary['num_errors']}"
    )
    print(f"[HEALTHCHECK] report={output_dir / 'preprocess_health_report.md'}")

    fail_on = str(args.fail_on)
    if fail_on == "warning" and (num_warnings > 0 or num_errors > 0):
        raise SystemExit(1)
    if fail_on == "error" and num_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
