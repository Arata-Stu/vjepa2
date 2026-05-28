from __future__ import annotations

import argparse
import csv
import json
import math
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


DEFAULT_METRICS = ("window_active_pixel_ratio", "window_activity_score")
DEFAULT_PERCENTILES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0)
DEFAULT_THRESHOLDS = (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to analyze H5 files. Install project dependencies first.")


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
    return sorted([p for p in dataset_root.glob(pattern) if p.is_file()])


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _infer_dataset_family(file_path: Path, representation: str) -> str:
    rep = str(representation).strip().lower()
    if rep == "event_voxel_grid_m3ed":
        return "m3ed"
    if rep == "event_voxel_grid_1mpx":
        return "1mpx"
    if rep == "event_voxel_grid_eventscape":
        return "eventscape"
    if rep == "event_voxel_grid":
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


def _read_1d_dataset(dset: Any, chunk_size: int) -> np.ndarray:
    n = int(dset.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    parts: list[np.ndarray] = []
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        arr = np.asarray(dset[start:end], dtype=np.float32).reshape(-1)
        if arr.size > 0:
            parts.append(arr)
    if len(parts) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def _metric_summary(values: np.ndarray, percentiles: list[float]) -> dict[str, Any]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {},
        }
    q = np.percentile(finite, percentiles)
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "percentiles": {f"p{p:g}": float(v) for p, v in zip(percentiles, q)},
    }


def _histogram_summary(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> dict[str, Any]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    counts, edges = np.histogram(finite, bins=int(bins), range=value_range)
    total = int(finite.size)
    fractions = counts.astype(np.float64) / float(total) if total > 0 else np.zeros_like(counts, dtype=np.float64)
    cdf = np.cumsum(fractions)
    return {
        "bin_edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
        "fractions": [float(v) for v in fractions.tolist()],
        "cdf": [float(v) for v in cdf.tolist()],
    }


def _threshold_rows(values: np.ndarray, thresholds: list[float], metric: str) -> list[dict[str, Any]]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    total = int(finite.size)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        keep = int(np.count_nonzero(finite >= float(threshold)))
        drop = total - keep
        rows.append(
            {
                "metric": metric,
                "threshold": float(threshold),
                "count_total": total,
                "count_keep_ge_threshold": keep,
                "count_drop_lt_threshold": drop,
                "keep_ratio_ge_threshold": float(keep) / float(total) if total > 0 else None,
                "drop_ratio_lt_threshold": float(drop) / float(total) if total > 0 else None,
            }
        )
    return rows


def _percentile_rows(values: np.ndarray, percentiles: list[float], metric: str) -> list[dict[str, Any]]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return []
    q = np.percentile(finite, percentiles)
    rows: list[dict[str, Any]] = []
    for p, value in zip(percentiles, q):
        drop_ratio = float(p) / 100.0
        rows.append(
            {
                "metric": metric,
                "percentile": float(p),
                "threshold_at_percentile": float(value),
                "drop_ratio_if_filter_lt_threshold": drop_ratio,
                "keep_ratio_if_filter_ge_threshold": 1.0 - drop_ratio,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_float(value: float | None, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(float(value), fmt)


def _build_histogram_svg(
    histograms: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    chart_title: str,
    value_range: tuple[float, float],
) -> None:
    metrics = list(histograms.keys())
    if len(metrics) == 0:
        return

    width = 1200
    panel_height = 280
    top_margin = 48
    left_margin = 72
    right_margin = 28
    bottom_margin = 56
    inner_gap = 26
    height = top_margin + len(metrics) * panel_height + (len(metrics) - 1) * inner_gap + 24
    plot_width = width - left_margin - right_margin
    plot_height = panel_height - bottom_margin - 46
    min_x, max_x = value_range
    palette = ["#2D6CDF", "#159570", "#C97A10", "#C43E3E"]

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left_margin}" y="28" font-size="24" font-family="monospace" fill="#111111">{_svg_escape(chart_title)}</text>',
    ]

    for metric_idx, metric in enumerate(metrics):
        panel_top = top_margin + metric_idx * (panel_height + inner_gap)
        plot_left = left_margin
        plot_top = panel_top + 30
        plot_bottom = plot_top + plot_height
        plot_right = plot_left + plot_width
        color = palette[metric_idx % len(palette)]
        hist = histograms[metric]
        summary = summaries[metric]
        counts = hist["counts"]
        edges = hist["bin_edges"]
        fractions = hist["fractions"]
        y_max = max(max(fractions), 1e-6)

        lines.append(
            f'<text x="{plot_left}" y="{panel_top + 18}" font-size="18" font-family="monospace" fill="#111111">{_svg_escape(metric)}</text>'
        )
        subtitle = (
            f"count={summary['count']}  mean={_format_float(summary['mean'])}  "
            f"p5={_format_float(summary['percentiles'].get('p5'))}  "
            f"p50={_format_float(summary['percentiles'].get('p50'))}  "
            f"p95={_format_float(summary['percentiles'].get('p95'))}"
        )
        lines.append(
            f'<text x="{plot_left}" y="{panel_top + 38}" font-size="13" font-family="monospace" fill="#444444">{_svg_escape(subtitle)}</text>'
        )

        lines.append(
            f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#dddddd" stroke-width="1"/>'
        )

        for tick_idx in range(6):
            frac = tick_idx / 5.0
            y = plot_bottom - frac * plot_height
            label = f"{frac * y_max:.3f}"
            lines.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#eeeeee" stroke-width="1"/>')
            lines.append(
                f'<text x="{plot_left - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="monospace" fill="#666666">{_svg_escape(label)}</text>'
            )

        for tick_idx in range(11):
            frac = tick_idx / 10.0
            x = plot_left + frac * plot_width
            label = min_x + frac * (max_x - min_x)
            lines.append(f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 6}" stroke="#888888" stroke-width="1"/>')
            lines.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 22}" font-size="11" text-anchor="middle" font-family="monospace" fill="#666666">{_svg_escape(f"{label:.2f}")}</text>'
            )

        for bin_idx, fraction in enumerate(fractions):
            x0 = plot_left + (edges[bin_idx] - min_x) / max(max_x - min_x, 1e-12) * plot_width
            x1 = plot_left + (edges[bin_idx + 1] - min_x) / max(max_x - min_x, 1e-12) * plot_width
            bar_width = max(x1 - x0 - 1.0, 0.2)
            bar_height = (float(fraction) / y_max) * plot_height if y_max > 0 else 0.0
            y0 = plot_bottom - bar_height
            lines.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" opacity="0.85"/>'
            )

        percentile_lines = [
            ("p5", "#B42318"),
            ("p25", "#F79009"),
            ("p50", "#101828"),
            ("p75", "#16A34A"),
            ("p95", "#7A5AF8"),
        ]
        legend_x = plot_right - 250
        legend_y = plot_top + 16
        for idx, (key, stroke) in enumerate(percentile_lines):
            value = summary["percentiles"].get(key)
            if value is None:
                continue
            x = plot_left + (float(value) - min_x) / max(max_x - min_x, 1e-12) * plot_width
            x = max(plot_left, min(plot_right, x))
            lines.append(
                f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{stroke}" stroke-width="2" stroke-dasharray="4 4"/>'
            )
            legend_entry_y = legend_y + idx * 16
            lines.append(
                f'<line x1="{legend_x}" y1="{legend_entry_y - 4}" x2="{legend_x + 18}" y2="{legend_entry_y - 4}" stroke="{stroke}" stroke-width="2" stroke-dasharray="4 4"/>'
            )
            lines.append(
                f'<text x="{legend_x + 24}" y="{legend_entry_y}" font-size="11" font-family="monospace" fill="#555555">{_svg_escape(f"{key}={float(value):.4f}")}</text>'
            )

        lines.append(
            f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{plot_bottom + 42}" font-size="12" text-anchor="middle" font-family="monospace" fill="#666666">ratio value</text>'
        )
        lines.append(
            f'<text transform="translate({plot_left - 54},{(plot_top + plot_bottom) / 2:.2f}) rotate(-90)" font-size="12" text-anchor="middle" font-family="monospace" fill="#666666">fraction of windows</text>'
        )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _analyze_file(
    file_path: Path,
    dataset_root: Path | None,
    metrics: list[str],
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _require_h5py()
    with h5py.File(str(file_path), "r") as h5f:
        if "voxels" not in h5f or h5f["voxels"].ndim != 4:
            raise ValueError("not a voxel H5 file")
        voxels = h5f["voxels"]
        n_samples = int(voxels.shape[0])
        representation = str(_safe_attr(h5f.attrs, "representation", ""))
        row: dict[str, Any] = {
            "file": str(file_path),
            "relative_file": _relative_or_name(file_path, dataset_root),
            "dataset_family": _infer_dataset_family(file_path=file_path, representation=representation),
            "representation": representation,
            "samples": n_samples,
            "shape": [int(v) for v in voxels.shape],
        }
        arrays: dict[str, np.ndarray] = {}
        for metric in metrics:
            exists = metric in h5f and len(h5f[metric]) >= n_samples
            row[f"{metric}_present"] = bool(exists)
            if not exists:
                row[f"{metric}_count"] = 0
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                continue

            values = _read_1d_dataset(h5f[metric], chunk_size=chunk_size)
            values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
            arrays[metric] = values
            row[f"{metric}_count"] = int(values.size)
            row[f"{metric}_min"] = None if values.size == 0 else float(values.min())
            row[f"{metric}_max"] = None if values.size == 0 else float(values.max())
            row[f"{metric}_mean"] = None if values.size == 0 else float(values.mean())
            row[f"{metric}_std"] = None if values.size == 0 else float(values.std())
        return row, arrays


def main() -> None:
    parser = argparse.ArgumentParser("Analyze dataset-wide activity metadata distributions from voxel H5 files.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single voxel H5 file.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory to scan for voxel H5 files.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --dataset_root for *.h5 files.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of files to analyze.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Scalar activity metadata keys to analyze.",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for ratio histograms.")
    parser.add_argument("--hist_min", type=float, default=0.0, help="Lower bound of histogram range.")
    parser.add_argument("--hist_max", type=float, default=1.0, help="Upper bound of histogram range.")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to report exactly from aggregated values.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Candidate thresholds. Report keep/drop ratios for value >= threshold.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=131072,
        help="Number of scalar metadata values to read per chunk from each H5 dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tmp/activity_distribution"),
        help="Output directory for summary artifacts.",
    )
    args = parser.parse_args()

    files = _select_files(
        input_path=args.input_path,
        dataset_root=args.dataset_root,
        recursive=bool(args.recursive),
    )
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise FileNotFoundError("no input files found")

    filtered_files = [p for p in files if _is_voxel_h5(p)]
    if len(filtered_files) == 0:
        raise FileNotFoundError("no voxel h5 files found (requires root dataset 'voxels' with 4D shape)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [str(m) for m in args.metrics]
    percentiles = sorted(set(float(v) for v in args.percentiles))
    thresholds = sorted(set(float(v) for v in args.thresholds))
    value_range = (float(args.hist_min), float(args.hist_max))
    if not value_range[0] < value_range[1]:
        raise ValueError("histogram range must satisfy hist_min < hist_max")

    dataset_root = args.dataset_root
    per_file_rows: list[dict[str, Any]] = []
    aggregated: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
    files_with_metric: dict[str, int] = {metric: 0 for metric in metrics}

    for file_path in filtered_files:
        row, arrays = _analyze_file(
            file_path=file_path,
            dataset_root=dataset_root,
            metrics=metrics,
            chunk_size=max(1, int(args.chunk_size)),
        )
        per_file_rows.append(row)
        for metric, values in arrays.items():
            aggregated[metric].append(values)
            files_with_metric[metric] += 1
        summary_tokens: list[str] = [f"samples={row['samples']}", f"dataset={row['dataset_family']}"]
        for metric in metrics:
            mean_value = row.get(f"{metric}_mean")
            if mean_value is not None:
                summary_tokens.append(f"{metric}_mean={float(mean_value):.6f}")
        print(f"[OK] {file_path} | " + " | ".join(summary_tokens))

    overall_metrics: dict[str, dict[str, Any]] = {}
    histogram_map: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []
    percentile_rows: list[dict[str, Any]] = []

    for metric in metrics:
        if len(aggregated[metric]) == 0:
            overall_metrics[metric] = {
                "files_with_metric": 0,
                "files_missing_metric": len(filtered_files),
                "summary": _metric_summary(np.empty((0,), dtype=np.float32), percentiles),
                "histogram": _histogram_summary(np.empty((0,), dtype=np.float32), bins=int(args.bins), value_range=value_range),
            }
            continue

        all_values = np.concatenate(aggregated[metric], axis=0) if len(aggregated[metric]) > 1 else aggregated[metric][0]
        summary = _metric_summary(all_values, percentiles)
        histogram = _histogram_summary(all_values, bins=int(args.bins), value_range=value_range)
        histogram_map[metric] = histogram
        threshold_rows.extend(_threshold_rows(all_values, thresholds=thresholds, metric=metric))
        percentile_rows.extend(_percentile_rows(all_values, percentiles=percentiles, metric=metric))
        overall_metrics[metric] = {
            "files_with_metric": int(files_with_metric[metric]),
            "files_missing_metric": int(len(filtered_files) - files_with_metric[metric]),
            "summary": summary,
            "histogram": histogram,
        }
        pct = summary["percentiles"]
        print(
            "[METRIC] "
            f"{metric} | count={summary['count']} | mean={_format_float(summary['mean'])} | "
            f"p1={_format_float(pct.get('p1'))} | p5={_format_float(pct.get('p5'))} | "
            f"p50={_format_float(pct.get('p50'))} | p95={_format_float(pct.get('p95'))} | "
            f"p99={_format_float(pct.get('p99'))}"
        )

    report = {
        "input_path": None if args.input_path is None else str(args.input_path),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "num_files": len(filtered_files),
        "metrics": metrics,
        "histogram_bins": int(args.bins),
        "histogram_range": [float(value_range[0]), float(value_range[1])],
        "percentiles": percentiles,
        "thresholds": thresholds,
        "overall": overall_metrics,
    }

    summary_json = output_dir / "activity_summary.json"
    per_file_csv = output_dir / "activity_per_file.csv"
    percentile_csv = output_dir / "activity_percentiles.csv"
    threshold_csv = output_dir / "activity_threshold_sweep.csv"
    histogram_svg = output_dir / "activity_histograms.svg"

    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    per_file_columns = [
        "file",
        "relative_file",
        "dataset_family",
        "representation",
        "samples",
        "shape",
    ]
    for metric in metrics:
        per_file_columns.extend(
            [
                f"{metric}_present",
                f"{metric}_count",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )
    _write_csv(path=per_file_csv, rows=per_file_rows, columns=per_file_columns)
    _write_csv(
        path=percentile_csv,
        rows=percentile_rows,
        columns=[
            "metric",
            "percentile",
            "threshold_at_percentile",
            "drop_ratio_if_filter_lt_threshold",
            "keep_ratio_if_filter_ge_threshold",
        ],
    )
    _write_csv(
        path=threshold_csv,
        rows=threshold_rows,
        columns=[
            "metric",
            "threshold",
            "count_total",
            "count_keep_ge_threshold",
            "count_drop_lt_threshold",
            "keep_ratio_ge_threshold",
            "drop_ratio_lt_threshold",
        ],
    )

    if len(histogram_map) > 0:
        _build_histogram_svg(
            histograms=histogram_map,
            summaries={metric: overall_metrics[metric]["summary"] for metric in histogram_map.keys()},
            output_path=histogram_svg,
            chart_title="Activity Distribution Histograms",
            value_range=value_range,
        )

        for metric, histogram in histogram_map.items():
            rows = []
            edges = histogram["bin_edges"]
            counts = histogram["counts"]
            fractions = histogram["fractions"]
            cdf = histogram["cdf"]
            for idx in range(len(counts)):
                rows.append(
                    {
                        "metric": metric,
                        "bin_index": idx,
                        "bin_start": edges[idx],
                        "bin_end": edges[idx + 1],
                        "count": counts[idx],
                        "fraction": fractions[idx],
                        "cdf": cdf[idx],
                    }
                )
            _write_csv(
                path=output_dir / f"{metric}_histogram.csv",
                rows=rows,
                columns=["metric", "bin_index", "bin_start", "bin_end", "count", "fraction", "cdf"],
            )

    print(
        "[SUMMARY] "
        f"files={len(filtered_files)} | summary_json={summary_json} | per_file_csv={per_file_csv} | "
        f"percentile_csv={percentile_csv} | threshold_csv={threshold_csv} | histogram_svg={histogram_svg}"
    )


if __name__ == "__main__":
    main()
