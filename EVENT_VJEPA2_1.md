# Event H5 V-JEPA2.1 Notes

This repo now contains the e-jepa event-camera preprocessing and H5 loading path in the root
project, with `tmp/` kept only as a reference copy.

## Preprocessing

The preprocessing entry point is:

```bash
python3 scripts/preprocess/run_preprocess.py dataset=dsec
```

Dataset presets live under `scripts/preprocess/conf/dataset/` and cover DSEC, M3ED, 1MPX, and
EventScape. The output H5 files store `/voxels` as `[N, C, H, W]` plus timestamp/activity metadata.

## V-JEPA2.1 Event Training

Event H5 training uses `data.dataset_type: EventDataset`.

Two input policies are supported:

- `voxel_time_mode: channels`
  - H5 window sequence is model time.
  - For `t_bins=10, split_polarity=true`: model input is `C=20, T=dataset_fpcs`.
  - This is closest to the original V-JEPA2.1 video treatment.

- `voxel_time_mode: temporal_bins`
  - One or more H5 windows are expanded over their voxel bins.
  - For `t_bins=10, split_polarity=true, source_window_fpcs=1`: model input is `C=2, T=10`.
  - This tests whether Conv3D/RoPE should directly see voxel-bin time.

Configs:

```text
configs/train_2_1/event/vitb16-h480w640-windows-t10.yaml
configs/train_2_1/event/vitb16-h480w640-bins-t10.yaml
configs/train_2_1/event/vitb16-h240w320-windows-t10.yaml
configs/train_2_1/event/vitb16-h240w320-bins-t10.yaml
```

Run from the repository root with module-style launch:

```bash
python -m app.main --fname configs/train_2_1/event/vitb16-h480w640-bins-t10.yaml --devices cuda:0
```

TensorBoard scalars are written by rank 0 under the run folder:

```bash
tensorboard --logdir /path/to/run/folder/tensorboard --port 6006 --bind_all
```

The 480x640 configs pad smaller datasets to `[480, 640]`. The 240x320 configs assume the H5 inputs
are already at half scale or smaller when `preserve_input_size: true`.

## Scheduler

For roughly 1000 H5 files, prefer update-count-based control:

```yaml
optimization:
  samples_per_epoch: 8000
  total_updates: 120000
  warmup_updates: 10000
```

This keeps LR/WD/EMA tied to optimizer updates, not accidentally to the number of H5 files.
