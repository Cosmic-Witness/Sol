# exp_032: bounded mask-loss workspace

This experiment changes loss execution, preserving YOLO11's mask objective and
pretrained head. It is an infrastructure improvement; an accuracy gain requires
training and canonical PQ evaluation. See [the architecture review](../../docs/architecture-review.md).

Install into an environment with a suitable PyTorch build (CUDA for training):

```bash
python -m pip install -r experiments/exp_032_chunked/requirements.txt
python -m pytest tests -q
```

Audit training annotations and measure the isolated loss before GPU training:

```bash
python -m scripts.audit_resolution --annotations data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --output results/exp_032_resolution_audit.json
python -m scripts.probe_mask_loss --size 256 --instances 40 --output results/exp_032_loss_probe_cpu.json
python -m scripts.probe_mask_loss --device cuda --size 2048 --instances 40 --output results/exp_032_loss_probe_cuda.json
```

Use the audit's maximum instance count in the GPU command. Ten positive anchors
per instance is an explicit stress assumption, not an observed assignment count.
Each implementation gets its own process and timeout. Failure records preserve
logs; a timeout or software error is not reported as OOM. `saved_storage_bytes`
counts unique storages retained by autograd after forward; it is not peak device
memory. CPU RSS includes libraries and allocator caching. CUDA peak covers the
isolated loss, **not the detector**. Synthetic stripes stress memory only.
`pass_seconds` reports the cold and warm forward/backward passes separately.
The checked-in CPU report shows 28 times less retained storage, 64% lower peak
working set and approximately 1.9 times the warm runtime. GPU results are absent.

Prepare data using the canonical converter (full competition train images needed):

```bash
python -m experiments.exp_002_yolo_seg.src.prepare_yolo --annotations data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --images data/train_images --output data/yolo_corrected
```

After a full-model probe verifies crowded batches, optimizer state and EMA
validation fit, run with an explicit checkpoint:

```bash
python -m experiments.exp_032_chunked.src.train --data data/yolo_corrected/data.yaml --weights data/checkpoints/exp002_best.pt --project experiments/exp_032_chunked/checkpoints --name native --imgsz 2048 --batch 1 --chunk-size 8 --epochs 50
```

Choose the epoch count from measured epoch duration, leaving room for validation
and export within the session. No `time` override, resolution fallback, automatic
optimizer, mosaic, or frozen backbone is used. Inspect
`native/resolved_experiment.json` before interpreting a run. Checkpoints are saved
each epoch. This entry point intentionally rejects an existing run directory and
does not implement resume; a new run from an explicit checkpoint starts a fresh
optimizer/schedule and must be described as a fine-tune.

Inference architecture is unchanged. Use the existing exp_005 native-resolution
PQ sweep and shared panoptic painting; resweep erosion because label correction
changes its optimum. Use exactly the selected rule for submission. Do not treat
Ultralytics validation mAP as competition PQ or submit without improved PQ.

Checkpoints contain the custom criterion, so retain the repository on Python's
import path when loading them. Install is opt-in: historical YOLO runs do not
change behavior when this file is added.
