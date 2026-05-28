from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


EVENT_GROUP_PATH = "prophesee/left"
SEMANTIC_TS_PATHS = {
    "semantics_ts": "semantics/ts",
    "semantics_ts_map": "semantics/ts_map_prophesee_left_t",
    "ovc_ts_map": "ovc/ts_map_prophesee_left_t",
}
DEPTH_TS_PATHS = {
    "depth_ts": "depth_gt/ts",
    "depth_ts_map_left_t": "depth_gt/ts_map_prophesee_left_t",
    "depth_ts_map_left": "depth_gt/ts_map_prophesee_left",
}
SEMANTIC_LABEL_CANDIDATES = (
    "semantics/class_id",
    "semantics/labels",
    "semantics/label",
    "semantics/data",
    "semantics/image",
)
DEPTH_LABEL_CANDIDATES = (
    "depth_gt/depth",
    "depth_gt/depth_map",
    "depth_gt/data",
    "depth_gt/image",
)


def _open_event_group(filehandle: h5py.File) -> h5py.Group:
    if EVENT_GROUP_PATH not in filehandle:
        raise KeyError(f"missing group '{EVENT_GROUP_PATH}'")
    return filehandle[EVENT_GROUP_PATH]


def _is_valid_event_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            events = _open_event_group(h5f)
            return all(k in events for k in ("x", "y", "t", "p"))
    except Exception:
        return False


def _read_t_offset(filehandle: h5py.File) -> int:
    if "t_offset" not in filehandle:
        return 0
    return int(filehandle["t_offset"][()])


def _get_event_bounds_us(filehandle: h5py.File) -> tuple[int, int, int]:
    events = _open_event_group(filehandle)
    t_ds = events["t"]
    n = int(len(t_ds))
    if n <= 0:
        raise ValueError("empty event stream")
    t_offset = _read_t_offset(filehandle)
    t_first = int(t_ds[0]) + t_offset
    t_last = int(t_ds[n - 1]) + t_offset
    return t_first, t_last, t_offset


def _load_optional_1d(filehandle: h5py.File, path: str) -> np.ndarray | None:
    if path not in filehandle:
        return None
    arr = np.asarray(filehandle[path][()])
    arr = np.atleast_1d(arr).reshape(-1)
    return arr.astype(np.int64, copy=False)


def _find_matching_label_dataset(
    filehandle: h5py.File,
    *,
    candidates: tuple[str, ...],
    length0: int,
) -> str | None:
    for path in candidates:
        if path not in filehandle:
            continue
        ds = filehandle[path]
        if isinstance(ds, h5py.Dataset) and ds.ndim >= 3 and int(ds.shape[0]) == int(length0):
            return path
    return None


def _in_range_ratio(values: np.ndarray, start_us: int, end_inclusive_us: int) -> float:
    if values.size == 0:
        return 0.0
    mask = (values >= int(start_us)) & (values <= int(end_inclusive_us))
    return float(np.count_nonzero(mask)) / float(values.size)


def _score_alignment(values: np.ndarray, start_us: int, end_inclusive_us: int) -> tuple[float, int]:
    ratio = _in_range_ratio(values, start_us, end_inclusive_us)
    edge = abs(int(values[0]) - int(start_us)) + abs(int(values[-1]) - int(end_inclusive_us))
    return ratio, -edge


def _analyze_time_source(
    raw: np.ndarray | None,
    *,
    t_first_us: int,
    t_last_us: int,
    t_offset_us: int,
) -> dict[str, Any]:
    if raw is None:
        return {
            "present": False,
            "length": 0,
            "raw_min": None,
            "raw_max": None,
            "default_ratio": None,
            "default_shift_mode": "",
            "default_divisor": 1,
            "best_ratio": None,
            "best_divisor": None,
            "best_shift_mode": "",
            "best_shift_us": None,
            "best_min": None,
            "best_max": None,
        }

    arr_raw = np.asarray(raw, dtype=np.int64).reshape(-1)
    if arr_raw.size == 0:
        return {
            "present": True,
            "length": 0,
            "raw_min": None,
            "raw_max": None,
            "default_ratio": 0.0,
            "default_shift_mode": "as_is",
            "default_divisor": 1,
            "best_ratio": 0.0,
            "best_divisor": 1,
            "best_shift_mode": "as_is",
            "best_shift_us": 0,
            "best_min": None,
            "best_max": None,
        }

    shift_candidates = [("as_is", 0)]
    if int(t_offset_us) != 0:
        shift_candidates += [("plus_t_offset", int(t_offset_us)), ("minus_t_offset", -int(t_offset_us))]
    divisor_candidates = (1, 1000, 1_000_000)

    best: dict[str, Any] | None = None
    for divisor in divisor_candidates:
        arr_div = arr_raw if int(divisor) == 1 else np.floor_divide(arr_raw, int(divisor)).astype(np.int64, copy=False)
        for shift_name, shift_us in shift_candidates:
            arr = arr_div + int(shift_us)
            score = _score_alignment(arr, t_first_us, t_last_us)
            candidate = {
                "ratio": float(score[0]),
                "edge_score": int(score[1]),
                "divisor": int(divisor),
                "shift_mode": str(shift_name),
                "shift_us": int(shift_us),
                "min": int(arr[0]),
                "max": int(arr[-1]),
            }
            if best is None or (candidate["ratio"], candidate["edge_score"]) > (best["ratio"], best["edge_score"]):
                best = candidate

    assert best is not None
    arr_default = arr_raw
    default_mode = "as_is"
    default_ratio = _in_range_ratio(arr_default, t_first_us, t_last_us)

    return {
        "present": True,
        "length": int(arr_raw.size),
        "raw_min": int(arr_raw[0]),
        "raw_max": int(arr_raw[-1]),
        "default_ratio": float(default_ratio),
        "default_shift_mode": default_mode,
        "default_divisor": 1,
        "best_ratio": float(best["ratio"]),
        "best_divisor": int(best["divisor"]),
        "best_shift_mode": str(best["shift_mode"]),
        "best_shift_us": int(best["shift_us"]),
        "best_min": int(best["min"]),
        "best_max": int(best["max"]),
    }


def _choose_default_auto_source(filehandle: h5py.File, *, source_paths: dict[str, str], ordered_keys: tuple[str, ...]) -> tuple[str | None, np.ndarray | None]:
    for key in ordered_keys:
        arr = _load_optional_1d(filehandle, source_paths[key])
        if arr is not None and arr.size > 0:
            return key, arr
    return None, None


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _find_m3ed_h5_files(dataset_root: Path) -> list[Path]:
    input_files: list[Path] = []
    for sequence_dir in sorted(dataset_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        for h5file in sorted(sequence_dir.glob("*.h5")):
            if h5file.is_file() and _is_valid_event_h5(h5file):
                input_files.append(h5file)
    return input_files


def main() -> None:
    parser = argparse.ArgumentParser("Inspect M3ED semantic/depth timestamp sources and likely unit mismatches.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root directory containing M3ED sequence folders.")
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/m3ed_timestamp_debug"), help="Directory for CSV/JSON outputs.")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of H5 files to inspect.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    files = _find_m3ed_h5_files(dataset_root)
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise FileNotFoundError(f"No valid M3ED event H5 files found under {dataset_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary = {
        "num_files": 0,
        "semantic_missing_all_sources": 0,
        "semantic_default_auto_source_missing": 0,
        "semantic_best_divisor_not_1": 0,
        "semantic_default_ratio_zero_but_best_positive": 0,
        "semantic_label_mismatch": 0,
        "depth_missing_all_sources": 0,
        "depth_best_divisor_not_1": 0,
    }

    for file_path in files:
        with h5py.File(str(file_path), "r") as h5f:
            t_first_us, t_last_us, t_offset_us = _get_event_bounds_us(h5f)
            semantic_raw = {key: _load_optional_1d(h5f, path) for key, path in SEMANTIC_TS_PATHS.items()}
            depth_raw = {key: _load_optional_1d(h5f, path) for key, path in DEPTH_TS_PATHS.items()}

            semantic_auto_key, semantic_auto_arr = _choose_default_auto_source(
                h5f,
                source_paths=SEMANTIC_TS_PATHS,
                ordered_keys=("semantics_ts", "semantics_ts_map", "ovc_ts_map"),
            )
            depth_auto_key, depth_auto_arr = _choose_default_auto_source(
                h5f,
                source_paths=DEPTH_TS_PATHS,
                ordered_keys=("depth_ts", "depth_ts_map_left_t", "depth_ts_map_left"),
            )

            semantic_stats = {key: _analyze_time_source(arr, t_first_us=t_first_us, t_last_us=t_last_us, t_offset_us=t_offset_us) for key, arr in semantic_raw.items()}
            depth_stats = {key: _analyze_time_source(arr, t_first_us=t_first_us, t_last_us=t_last_us, t_offset_us=t_offset_us) for key, arr in depth_raw.items()}

            semantic_present_any = any(arr is not None and arr.size > 0 for arr in semantic_raw.values())
            depth_present_any = any(arr is not None and arr.size > 0 for arr in depth_raw.values())
            if not semantic_present_any:
                summary["semantic_missing_all_sources"] += 1
            if semantic_auto_key is None:
                summary["semantic_default_auto_source_missing"] += 1
            if not depth_present_any:
                summary["depth_missing_all_sources"] += 1

            semantic_default_ratio = None
            semantic_best_source = ""
            semantic_best_ratio = -1.0
            semantic_best_divisor = None
            semantic_best_shift_mode = ""
            semantic_best_len = 0
            for key, stats in semantic_stats.items():
                ratio = -1.0 if stats["best_ratio"] is None else float(stats["best_ratio"])
                if ratio > semantic_best_ratio:
                    semantic_best_ratio = ratio
                    semantic_best_source = key
                    semantic_best_divisor = stats["best_divisor"]
                    semantic_best_shift_mode = stats["best_shift_mode"]
                    semantic_best_len = int(stats["length"])
            if semantic_auto_key is not None:
                semantic_default_ratio = float(semantic_stats[semantic_auto_key]["default_ratio"])
            if semantic_best_divisor not in (None, 1):
                summary["semantic_best_divisor_not_1"] += 1
            if semantic_default_ratio is not None and semantic_default_ratio == 0.0 and semantic_best_ratio > 0.0:
                summary["semantic_default_ratio_zero_but_best_positive"] += 1

            depth_best_divisor = None
            depth_best_ratio = -1.0
            depth_best_source = ""
            for key, stats in depth_stats.items():
                ratio = -1.0 if stats["best_ratio"] is None else float(stats["best_ratio"])
                if ratio > depth_best_ratio:
                    depth_best_ratio = ratio
                    depth_best_divisor = stats["best_divisor"]
                    depth_best_source = key
            if depth_best_divisor not in (None, 1):
                summary["depth_best_divisor_not_1"] += 1

            semantic_label_default_match = False
            semantic_label_best_match = False
            semantic_label_default_path = ""
            semantic_label_best_path = ""
            if semantic_auto_arr is not None and semantic_auto_arr.size > 0:
                p = _find_matching_label_dataset(h5f, candidates=SEMANTIC_LABEL_CANDIDATES, length0=int(semantic_auto_arr.size))
                if p is not None:
                    semantic_label_default_match = True
                    semantic_label_default_path = p
            if semantic_best_source != "" and semantic_best_len > 0:
                p = _find_matching_label_dataset(h5f, candidates=SEMANTIC_LABEL_CANDIDATES, length0=int(semantic_best_len))
                if p is not None:
                    semantic_label_best_match = True
                    semantic_label_best_path = p
            if semantic_present_any and not semantic_label_best_match:
                summary["semantic_label_mismatch"] += 1

            row = {
                "file": str(file_path),
                "relative_file": _relative_to(file_path, dataset_root),
                "sequence": file_path.parent.name,
                "event_t_first_us": int(t_first_us),
                "event_t_last_us": int(t_last_us),
                "event_duration_s": float(max(0, t_last_us - t_first_us)) / 1e6,
                "t_offset_us": int(t_offset_us),
                "semantic_present_any": bool(semantic_present_any),
                "semantic_default_auto_source": "" if semantic_auto_key is None else semantic_auto_key,
                "semantic_default_auto_ratio": semantic_default_ratio,
                "semantic_best_source": semantic_best_source,
                "semantic_best_ratio": None if semantic_best_ratio < 0 else float(semantic_best_ratio),
                "semantic_best_divisor": semantic_best_divisor,
                "semantic_best_shift_mode": semantic_best_shift_mode,
                "semantic_label_default_match": bool(semantic_label_default_match),
                "semantic_label_default_path": semantic_label_default_path,
                "semantic_label_best_match": bool(semantic_label_best_match),
                "semantic_label_best_path": semantic_label_best_path,
                "depth_present_any": bool(depth_present_any),
                "depth_default_auto_source": "" if depth_auto_key is None else depth_auto_key,
                "depth_best_source": depth_best_source,
                "depth_best_ratio": None if depth_best_ratio < 0 else float(depth_best_ratio),
                "depth_best_divisor": depth_best_divisor,
            }

            for key, stats in semantic_stats.items():
                prefix = f"semantic_{key}"
                row[f"{prefix}_present"] = bool(stats["present"])
                row[f"{prefix}_length"] = int(stats["length"])
                row[f"{prefix}_raw_min"] = stats["raw_min"]
                row[f"{prefix}_raw_max"] = stats["raw_max"]
                row[f"{prefix}_default_ratio"] = stats["default_ratio"]
                row[f"{prefix}_best_ratio"] = stats["best_ratio"]
                row[f"{prefix}_best_divisor"] = stats["best_divisor"]
                row[f"{prefix}_best_shift_mode"] = stats["best_shift_mode"]

            for key, stats in depth_stats.items():
                prefix = f"depth_{key}"
                row[f"{prefix}_present"] = bool(stats["present"])
                row[f"{prefix}_length"] = int(stats["length"])
                row[f"{prefix}_raw_min"] = stats["raw_min"]
                row[f"{prefix}_raw_max"] = stats["raw_max"]
                row[f"{prefix}_default_ratio"] = stats["default_ratio"]
                row[f"{prefix}_best_ratio"] = stats["best_ratio"]
                row[f"{prefix}_best_divisor"] = stats["best_divisor"]
                row[f"{prefix}_best_shift_mode"] = stats["best_shift_mode"]

            rows.append(row)
            summary["num_files"] += 1

            print(
                "[OK] "
                f"{file_path} | semantic_present={semantic_present_any} | "
                f"semantic_auto={semantic_auto_key or 'missing'} | semantic_best={semantic_best_source or 'missing'} "
                f"(ratio={semantic_best_ratio:.3f}, divisor={semantic_best_divisor}) | "
                f"depth_best={depth_best_source or 'missing'} (ratio={depth_best_ratio:.3f}, divisor={depth_best_divisor})"
            )

    csv_path = output_dir / "m3ed_timestamp_debug.csv"
    json_path = output_dir / "m3ed_timestamp_summary.json"

    columns = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})

    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"[SUMMARY] files={summary['num_files']} | csv={csv_path} | json={json_path}")


if __name__ == "__main__":
    main()
