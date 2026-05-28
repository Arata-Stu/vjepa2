from __future__ import annotations

from pathlib import Path
import re
import numpy as np


def _compression_opts(
    compression_level: int = 1,
    shuffle: int = 2,
    compressor_type: int = 5,
) -> tuple[int, int, int, int, int, int, int]:
    level = int(compression_level)
    if level < 0 or level > 9:
        raise ValueError(f"compression_level must be in [0,9], got {compression_level}")
    if int(shuffle) not in (0, 1, 2):
        raise ValueError(f"shuffle must be one of {{0,1,2}}, got {shuffle}")
    return 0, 0, 0, 0, level, int(shuffle), int(compressor_type)


def get_h5_compression_flags(
    compression_level: int = 1,
    shuffle: int = 2,
    compressor_type: int = 5,
) -> dict:
    level = int(compression_level)
    if level < 0 or level > 9:
        raise ValueError(f"compression_level must be in [0,9], got {compression_level}")
    try:
        import hdf5plugin  # noqa: F401

        _ = hdf5plugin
        return dict(
            compression=32001,
            compression_opts=_compression_opts(
                compression_level=level,
                shuffle=shuffle,
                compressor_type=compressor_type,
            ),
        )
    except ImportError:
        return dict(
            compression="gzip",
            compression_opts=level,
        )


def tmp_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.name}{tmp_suffix}")


def tmp_media_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{tmp_suffix}{output_path.suffix}")


def cleanup_tmp_file(tmp_path: Path, context: str, strict: bool = True) -> bool:
    if not tmp_path.exists():
        return True

    try:
        tmp_path.unlink()
        print(f"[RESUME] removed stale tmp: {tmp_path}")
        return True
    except Exception as exc:
        msg = f"[FAILED] could not remove tmp file ({context}): {tmp_path} ({exc})"
        if strict:
            raise RuntimeError(msg) from exc
        print(msg)
        return False


def normalized_output_suffix(output_suffix: str) -> str:
    suffix = output_suffix
    if not suffix.endswith(".h5"):
        suffix = f"{suffix}.h5"
    return suffix


def normalized_output_subdir(output_subdir: str | None) -> str | None:
    if output_subdir is None:
        return None

    subdir = output_subdir.strip().strip("/\\")
    if len(subdir) == 0:
        return None

    subdir_path = Path(subdir)
    if subdir_path.is_absolute() or any(part in {".", ".."} for part in subdir_path.parts):
        raise ValueError("--output_subdir must be a relative path without '.' or '..'")
    return str(subdir_path)


def ensure_scale_tag_in_filename(filename: str, downsample_factor: int) -> str:
    factor = int(downsample_factor)
    if factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    tag = f"{factor}x"
    file_path = Path(filename)
    stem = file_path.stem
    suffix = file_path.suffix

    match = re.search(r"(^|[_-])(\d+x)(?=($|[_-]))", stem)
    if match is None:
        return f"{stem}_{tag}{suffix}"

    current = match.group(2)
    if current == tag:
        return filename

    start = match.start(2)
    end = match.end(2)
    new_stem = f"{stem[:start]}{tag}{stem[end:]}"
    return f"{new_stem}{suffix}"


def normalize_polarity_to_binary(
    polarity: np.ndarray,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """
    Normalize event polarity to binary convention {0,1}.

    Supports common source conventions:
    - signed {-1,+1}
    - binary {0,1}
    Any value > 0 becomes 1, otherwise 0.
    """
    arr = np.asarray(polarity)
    bin_arr = arr > 0
    out_dtype = np.uint8 if dtype is None else dtype
    return bin_arr.astype(out_dtype, copy=False)


class RgbMp4Writer:
    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        width: int,
        height: int,
    ):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for MP4 export. Install via `pip install opencv-python`.") from exc

        self._cv2 = cv2
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(fps, 1e-6)),
            (int(width), int(height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {self.output_path}")

    def write_rgb(self, frame_rgb: np.ndarray) -> None:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame_rgb must be HxWx3, got shape={frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = np.round(frame * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()
            self._writer = None


def infer_mp4_fps_from_timestamps_us(
    timestamps_us: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[float | None, str]:
    timestamps = np.asarray(timestamps_us, dtype=np.int64).reshape(-1)
    if timestamps.size < 2:
        return None, "insufficient_timestamps"

    deltas_us = np.diff(timestamps)
    positive_deltas_us = deltas_us[deltas_us > 0]
    if positive_deltas_us.size == 0:
        return None, "nonpositive_timestamp_deltas"

    median_delta_us = float(np.median(positive_deltas_us))
    if median_delta_us <= 0.0:
        return None, "invalid_median_timestamp_delta"
    return float(1_000_000.0 / median_delta_us), "inferred_median_timestamp_delta_us"


def resolve_mp4_fps(
    requested_fps: float | None,
    timestamps_us: np.ndarray | list[int] | tuple[int, ...],
    *,
    fallback_fps: float = 10.0,
) -> tuple[float, str]:
    if requested_fps is not None and float(requested_fps) > 0.0:
        return float(requested_fps), "explicit"

    inferred_fps, inferred_source = infer_mp4_fps_from_timestamps_us(timestamps_us)
    if inferred_fps is not None:
        return inferred_fps, inferred_source

    fallback = float(fallback_fps)
    if fallback <= 0.0:
        raise ValueError("fallback_fps must be > 0")
    return fallback, f"fallback:{inferred_source}"


class LazyRgbMp4Writer:
    def __init__(
        self,
        output_path: Path,
        *,
        fps: float | None,
        fallback_fps: float = 10.0,
    ):
        self.output_path = Path(output_path)
        self.requested_fps = None if fps is None else float(fps)
        if self.requested_fps is not None and self.requested_fps <= 0.0:
            self.requested_fps = None
        self.fallback_fps = float(fallback_fps)
        if self.fallback_fps <= 0.0:
            raise ValueError("fallback_fps must be > 0")

        self.resolved_fps: float | None = None
        self.fps_source: str = ""
        self._writer: RgbMp4Writer | None = None
        self._pending_frames: list[np.ndarray] = []
        self._pending_timestamps_us: list[int] = []

    def _flush_pending_frames(self) -> None:
        if self._writer is None:
            return
        for frame_rgb in self._pending_frames:
            self._writer.write_rgb(frame_rgb)
        self._pending_frames.clear()

    def _open_writer(self, fps: float, fps_source: str) -> None:
        if self._writer is not None:
            return
        if len(self._pending_frames) == 0:
            raise RuntimeError("cannot open MP4 writer without any pending frames")
        first_frame = np.asarray(self._pending_frames[0])
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError(f"frame_rgb must be HxWx3, got shape={first_frame.shape}")
        self._writer = RgbMp4Writer(
            self.output_path,
            fps=float(fps),
            width=int(first_frame.shape[1]),
            height=int(first_frame.shape[0]),
        )
        self.resolved_fps = float(fps)
        self.fps_source = str(fps_source)
        self._flush_pending_frames()

    def _try_open_writer(self) -> None:
        if self._writer is not None or len(self._pending_frames) == 0:
            return
        if self.requested_fps is not None:
            self._open_writer(fps=float(self.requested_fps), fps_source="explicit")
            return
        inferred_fps, inferred_source = infer_mp4_fps_from_timestamps_us(self._pending_timestamps_us)
        if inferred_fps is None:
            return
        self._open_writer(fps=inferred_fps, fps_source=inferred_source)

    def write_rgb(self, frame_rgb: np.ndarray, *, timestamp_us: int | None = None) -> None:
        frame = np.asarray(frame_rgb)
        if self._writer is not None:
            self._writer.write_rgb(frame)
            return

        self._pending_frames.append(frame)
        if timestamp_us is not None:
            self._pending_timestamps_us.append(int(timestamp_us))
        self._try_open_writer()

    def close(self) -> None:
        if self._writer is None and len(self._pending_frames) > 0:
            fps, fps_source = resolve_mp4_fps(
                requested_fps=self.requested_fps,
                timestamps_us=self._pending_timestamps_us,
                fallback_fps=self.fallback_fps,
            )
            self._open_writer(fps=fps, fps_source=fps_source)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
