# Preprocess Commands

基本は `dataset_root + output_root` で一括処理します。  
Hydra ランナー: `scripts/preprocess/run_preprocess.py`

デフォルト:

- `t_bins=10`
- `downsample_factor`: `dsec=1`, `eventscape=1`, `1mpx=2`, `m3ed=2`
- 極性は `{0,1}` と `{-1,+1}` の両方を受け付け、内部で `{0,1}` に正規化

## Local Presets

履歴ベースでそのまま叩ける Hydra preset を追加:

```bash
python3 scripts/preprocess/run_preprocess.py dataset=1mpx_t10
python3 scripts/preprocess/run_preprocess.py dataset=1mpx_t1
python3 scripts/preprocess/run_preprocess.py dataset=1mpx_event_image
python3 scripts/preprocess/run_preprocess.py dataset=eventscape_t10
python3 scripts/preprocess/run_preprocess.py dataset=eventscape_t1
python3 scripts/preprocess/run_preprocess.py dataset=eventscape_event_image
python3 scripts/preprocess/run_preprocess.py dataset=dsec_semantic_t10
python3 scripts/preprocess/run_preprocess.py dataset=dsec_semantic_t1
python3 scripts/preprocess/run_preprocess.py dataset=dsec_semantic_event_image
python3 scripts/preprocess/run_preprocess.py dataset=m3ed_semantic_t10
python3 scripts/preprocess/run_preprocess.py dataset=m3ed_semantic_t1
python3 scripts/preprocess/run_preprocess.py dataset=m3ed_semantic_event_image
```

Preset file:

- `scripts/preprocess/conf/dataset/1mpx_t10.yaml`
- `scripts/preprocess/conf/dataset/1mpx_t1.yaml`
- `scripts/preprocess/conf/dataset/1mpx_event_image.yaml`
- `scripts/preprocess/conf/dataset/eventscape_t10.yaml`
- `scripts/preprocess/conf/dataset/eventscape_t1.yaml`
- `scripts/preprocess/conf/dataset/eventscape_event_image.yaml`
- `scripts/preprocess/conf/dataset/dsec_semantic_t10.yaml`
- `scripts/preprocess/conf/dataset/dsec_semantic_t1.yaml`
- `scripts/preprocess/conf/dataset/dsec_semantic_event_image.yaml`
- `scripts/preprocess/conf/dataset/m3ed_semantic_t10.yaml`
- `scripts/preprocess/conf/dataset/m3ed_semantic_t1.yaml`
- `scripts/preprocess/conf/dataset/m3ed_semantic_event_image.yaml`

`dsec` / `m3ed` は label 名に合わせて preset 名と出力 dir 名を `semantic` で統一しています。

`*_event_image` preset は `representation=event_image` と `save_mp4=true` を含みます。event image は
window 全体を 1 枚の RGB 画像に積算するので、preset 名は `t10` ではなく `event_image` を軸にしています。
さらに `split_chunk_duration_s` が有効な preset では、companion `mp4` も H5 chunk と同じ suffix
（例: `_part0000.mp4`）で自動 split されます。

## DSEC

想定例: `dsec/train/<sequence>/events/left/events.h5`

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels \
  dataset.splits=[train,test]
```

前処理と同時に 20 秒 chunk も作る:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels_seg \
  dataset.window_mode=image_middle \
  dataset.sync_segmentation=true \
  dataset.segmentation_subdir=11classes \
  dataset.segmentation_tolerance_us=0 \
  dataset.split_chunk_duration_s=20 \
  dataset.split_output_root=/data/DSEC_voxels_seg_20s \
  dataset.split_delete_source_after_success=true
```

image middle:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels \
  dataset.window_mode=image_middle
```

image middle + segmentation sync:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels_seg \
  dataset.window_mode=image_middle \
  dataset.sync_segmentation=true \
  dataset.segmentation_root=/data/DSEC \
  dataset.segmentation_subdir=11classes \
  dataset.segmentation_tolerance_us=0
```

`image_root` は通常不要です（`<sequence>/images/...` を自動参照）。  
画像ディレクトリが別ルートに分離されている場合のみ `dataset.image_root=...` を指定します。

`dataset.sync_segmentation=true` のときは、同期 metadata だけでなく `embedded_segmentation` も同じ H5 に保存します。  
さらに `activity_mode=full|light` で `window_activity_*` metadata も保存されます。

## 1MPX

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels
```

前処理と同時に 20 秒 chunk も作る:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels \
  dataset.split_chunk_duration_s=20 \
  dataset.split_output_root=/data/1mpx_voxels_20s \
  dataset.split_delete_source_after_success=true
```

split 指定:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels \
  dataset.splits=[train,test,val]
```

## M3ED

event + semantic:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.window_mode=semantics_middle \
  dataset.output_suffix=_voxels_semantic.h5
```

`semantics_middle` は既定で、現在 semantic 対応が確認できている M3ED sequence のみを処理します。
既知 subset 以外も無理に流したい場合は `dataset.filter_known_semantic_sequences=false` を追加してください。

前処理と同時に 20 秒 chunk も作る:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.window_mode=semantics_middle \
  dataset.output_suffix=_voxels_semantic.h5 \
  dataset.activity_mode=full \
  dataset.split_chunk_duration_s=20 \
  dataset.split_output_root=/data/m3ed_voxels_semantic_20s \
  dataset.split_delete_source_after_success=true
```

event + depth:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_depth \
  dataset.window_mode=depth_middle \
  dataset.output_suffix=_voxels_depth.h5
```

`depth_middle` も既定で、現在 depth 対応が確認できている M3ED sequence のみを処理します。
既知 subset 以外も無理に流したい場合は `dataset.filter_known_depth_sequences=false` を追加してください。

preprocessed H5 の voxel と semantic label を並べて目視確認:

```bash
python3 scripts/preprocess/visualize_m3ed_sync_debug.py \
  --dataset_root /data/m3ed_voxels_semantic_20s \
  --target semantic \
  --output_dir /tmp/m3ed_semantic_debug \
  --num_samples 4 \
  --selection_mode valid
```

depth 版:

```bash
python3 scripts/preprocess/visualize_m3ed_sync_debug.py \
  --dataset_root /data/m3ed_voxels_depth_20s \
  --target depth \
  --output_dir /tmp/m3ed_depth_debug \
  --num_samples 4 \
  --selection_mode valid
```

各 preview には `activity / polarity / label / overlay` を並べて保存します。  
`selection_mode invalid` にすると、label を引けない window を重点的に確認できます。  
raw H5 側の timestamp source 候補や単位ズレ調査は `scripts/preprocess/debug_m3ed_timestamps.py` を使ってください。

## EventScape

想定例: `event_scape/Town05_test/<sequence>/events/data/*.npz`

全split一括:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels
```

`dsec` / `eventscape` も `activity` メタデータを保存します。`dataset.activity_mode=full|light` で切り替えでき、デフォルトは `full` です。

Town系 split を指定:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels_town05 \
  'dataset.splits=[Town05_test,Town05_val]'
```

Town01-03 train のみ:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels_train \
  'dataset.splits=[Town01-03_train]'
```

## 共通上書き例

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.num_processes=8 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.accum_time=50000 \
  dataset.output_dtype=float16
```

## Event Image Presets

event image + companion MP4 を出したい場合は、次の preset をそのまま使えます:

```bash
python3 scripts/preprocess/run_preprocess.py dataset=eventscape_event_image
python3 scripts/preprocess/run_preprocess.py dataset=dsec_semantic_event_image
python3 scripts/preprocess/run_preprocess.py dataset=m3ed_semantic_event_image
python3 scripts/preprocess/run_preprocess.py dataset=1mpx_event_image
```

`dsec_semantic_event_image` は既定で `dataset.segmentation_tolerance_us=5000` を含みます。  
`mp4_fps` は省略時に timestamp 間隔から自動推定されます。
既に split H5 が存在する場合でも、source H5 と companion mp4 が残っていれば rerun で不足している split mp4 を補完できます。

## 解析コマンド（voxel H5）

dataset_root 一括解析 + 可視化:

```bash
python3 scripts/preprocess/analyze_voxel_h5.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_analysis \
  --polarity_order negpos \
  --vote_use_abs_for_split_polarity \
  --write_mp4 \
  --mp4_fps 20
```

長尺を制限したい場合:

```bash
python3 scripts/preprocess/analyze_voxel_h5.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_analysis \
  --write_mp4 \
  --mp4_fps 30 \
  --mp4_max_frames 600
```

単一ファイル:

```bash
python3 scripts/preprocess/analyze_voxel_h5.py \
  --input_path /data/preprocessed_voxels/sample_voxels_2x.h5 \
  --output_dir /data/preprocessed_voxels_analysis_single
```

activity 指標の全体分布を見て閾値設計したい場合:

```bash
python3 scripts/preprocess/analyze_activity_distribution.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_activity \
  --bins 100 \
  --thresholds 0.0 0.001 0.005 0.01 0.02 0.05 0.1
```

主な出力:

- `activity_histograms.svg`: `window_active_pixel_ratio` / `window_activity_score` の正規化ヒストグラム
- `activity_percentiles.csv`: 例えば下位 5%, 10%, 25% を落とすときの閾値候補
- `activity_threshold_sweep.csv`: 指定閾値ごとの keep/drop 割合
- `activity_per_file.csv`: sequence/H5 ごとの平均値や外れ値確認用

filter 後に残る clip / 落ちる clip を見たい場合:

```bash
python3 scripts/preprocess/visualize_activity_filter.py \
  --dataset_root /data/m3ed_voxels \
  --frames_per_clip 16 \
  --min_clip_mean_active 0.01 \
  --min_clip_active_frac 0.25 \
  --output_dir /data/m3ed_filter_preview
```

主な出力:

- `clip_filter_histograms.svg`: clip 単位で見た keep/drop 分布
- `examples/keep_random/contact_sheet.png`: 典型的に残る clip
- `examples/drop_random/contact_sheet.png`: 典型的に落ちる clip
- `examples/keep_boundary/contact_sheet.png`: しきい値ギリギリで残る clip
- `examples/drop_boundary/contact_sheet.png`: しきい値ギリギリで落ちる clip
- `clip_filter_rows.csv`: clip ごとの keep/drop 判定

前処理直後に構造エラーや怪しい sequence をまとめて見たい場合:

```bash
python3 scripts/preprocess/validate_preprocessed_h5.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_healthcheck \
  --fail_on error
```

低 activity の outlier が特定 sequence に偏っていたら warning でも止めたい場合:

```bash
python3 scripts/preprocess/validate_preprocessed_h5.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_healthcheck \
  --fail_on warning
```

主な出力:

- `preprocess_health_report.md`: 最初に見るべき summary report
- `preprocess_health_rows.csv`: file ごとの event/activity 統計と status
- `preprocess_health_issues.csv`: error / warning 一覧
- `preprocess_health_summary.json`: CI 向け summary

この healthcheck は、例えば次を自動で拾います。

- `voxels` や `window_event_count` など必須 dataset の欠落
- window 時刻の非単調や `end <= start`
- NaN / Inf voxel
- `semantic` / `depth` 同期 H5 なのに label reference が無い
- M3ED semantic が `ovc_ts_map` fallback に落ちている
- zero-event window が極端に多い file
- low activity outlier が特定 sequence に偏っている

## 20秒分割（事前学習向け）

M3ED など長尺 voxel H5 を、約20秒ごとの複数 H5 に分割:

```bash
python3 scripts/preprocess/split_voxel_h5_by_duration.py \
  --dataset_root /data/m3ed_voxels_semantic \
  --output_root /data/m3ed_voxels_semantic_20s \
  --chunk_duration_s 20 \
  --num_processes 8 \
  --copy_batch_size 8
```

copy中の進捗ログを見たい場合（N秒ごとに表示）:

```bash
python3 scripts/preprocess/split_voxel_h5_by_duration.py \
  --dataset_root /data/m3ed_voxels_semantic \
  --output_root /data/m3ed_voxels_semantic_20s \
  --chunk_duration_s 20 \
  --num_processes 1 \
  --copy_batch_size 8 \
  --progress_interval_s 10 \
  --log_chunk_progress
```

`num_processes>1` の場合でもログは親プロセスで集約表示され、worker間で混線しにくい形式になります。

`dsec` / `m3ed` / `1mpx` では `dataset.split_chunk_duration_s=20` を `run_preprocess.py` に渡すと、各 voxel H5 を書いた直後に自動分割できます。`dataset.split_delete_source_after_success=true` を付けると、分割成功後に長尺の元 H5 を削除して二重保管を避けられます。

activity メタデータは `dataset.activity_mode=full|light` で切り替えられます。デフォルトは `full` で、`full` は時空間 token-grid、`light` は空間 grid のみを保存します。

`m3ed` の `semantics_middle` / `depth_middle` では、対応する annotation も前処理 H5 に埋め込まれます。これにより、split 後 H5 だけでも downstream 学習を完結できます。

DSEC + semantic 同期済みなどで高速化したい場合:

```bash
python3 scripts/preprocess/split_voxel_h5_by_duration.py \
  --dataset_root /data/DSEC_voxels_seg \
  --output_root /data/DSEC_voxels_seg_20s \
  --chunk_duration_s 20 \
  --num_processes 2 \
  --copy_batch_size 32 \
  --metadata_mode minimal
```

## DSEC 同期デバッグ

raw events と image timestamps の単位/整合を確認:

```bash
python3 scripts/preprocess/debug_dsec_preprocess.py \
  --dataset_root /data/DSEC \
  --split test \
  --sequence interlaken_00_a \
  --divisors 1 1000
```

preprocessed 出力と突き合わせ:

```bash
python3 scripts/preprocess/debug_dsec_preprocess.py \
  --dataset_root /data/DSEC \
  --split test \
  --sequence interlaken_00_a \
  --preprocessed_h5 /data/DSEC_voxels_seg/test/interlaken_00_a/events/left/events_voxels_1x.h5 \
  --divisors 1 1000
```

preprocessed H5 の voxel と semantic label を並べて目視確認:

```bash
python3 scripts/preprocess/visualize_dsec_semantic_debug.py \
  --dataset_root /data/DSEC_voxels_seg_val \
  --output_dir /tmp/dsec_semantic_debug \
  --num_samples 4 \
  --selection_mode available
```

各 preview には `activity / polarity / label / overlay` を並べて保存します。  
`selection_mode missing` にすると、`segmentation_available=0` の window を重点的に確認できます。
