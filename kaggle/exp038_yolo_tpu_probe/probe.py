"""Probe real YOLO segmentation optimizer steps on TPU; GPU disabled."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PJRT_DEVICE'] = 'TPU'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import torch
    from ultralytics.models.yolo.segment import SegmentationTrainer
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh
    torch.manual_seed(2026)
    xr.use_spmd()
    count = xr.global_runtime_device_count()
    device = xm.xla_device()
    trainer = SegmentationTrainer(overrides=dict(
        model=args.weights, data=args.data, device='cpu', imgsz=args.size,
        batch=8, epochs=1, mask_ratio=args.mask_ratio, overlap_mask=True,
        optimizer='AdamW', lr0=1e-5, workers=0, plots=False, cache=False,
        mosaic=0., mixup=0., copy_paste=0., degrees=0., hsv_h=0., hsv_s=0.,
        hsv_v=0., fliplr=0., flipud=0., project='/kaggle/working/setup',
        name=f'{args.size}_{args.mask_ratio}'))
    trainer._setup_train()
    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    model = trainer.model.to(device).train()
    # Recreate the criterion after moving the model so its device is TPU.
    model.criterion = model.init_criterion()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5)
    mesh = Mesh(np.arange(count), (count, 1, 1, 1), ('data', 'c', 'h', 'w'))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    xs.mark_sharding(batch['img'], mesh, ('data', 'c', 'h', 'w'))
    timings, losses = [], []
    for _ in range(2):
        start = time.perf_counter()
        optimiser.zero_grad()
        loss, components = model(batch)
        loss.sum().backward()
        optimiser.step()
        xm.mark_step()
        values = components.cpu()
        timings.append(time.perf_counter()-start)
        losses.append([float(x) for x in values])
    report = dict(status='complete', size=args.size, mask_ratio=args.mask_ratio,
                  global_batch=8, devices=count, seconds=timings, losses=losses,
                  instances=int(batch['cls'].shape[0]), mask_shape=list(batch['masks'].shape),
                  finite=all(np.isfinite(losses).flat), torch=torch.__version__,
                  note='Two real optimizer steps; feasibility only, no quality claim.')
    Path(f'/kaggle/working/yolo_{args.size}_{args.mask_ratio}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data'); parser.add_argument('--weights')
    parser.add_argument('--size', type=int); parser.add_argument('--mask-ratio', type=int)
    args = parser.parse_args()
    if args.size:
        worker(args); return
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.0'])
    root = next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')).parent
    annotations, images = root/'MAGFiLO_1.0_Annotations_kaggle2026_train.json', root/'train_images'
    weights = list(Path('/kaggle/input').rglob('checkpoints/best.pt'))
    if len(weights) != 1: raise RuntimeError(weights)
    prepared = Path('/kaggle/working/yolo_data')
    subprocess.check_call([sys.executable, '-m', 'experiments.exp_002_yolo_seg.src.prepare_yolo',
                           '--annotations', str(annotations), '--images', str(images),
                           '--output', str(prepared), '--val-fraction', '.15'])
    summary = []
    for size, ratio in [(1024, 2), (1024, 1), (2048, 1)]:
        log = Path(f'/kaggle/working/yolo_{size}_{ratio}.log')
        with log.open('w') as stream:
            try:
                run = subprocess.run([sys.executable, __file__, '--data', str(prepared/'data.yaml'),
                                      '--weights', str(weights[0]), '--size', str(size),
                                      '--mask-ratio', str(ratio)], stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=1000)
                summary.append(dict(size=size, mask_ratio=ratio, returncode=run.returncode))
            except subprocess.TimeoutExpired:
                summary.append(dict(size=size, mask_ratio=ratio, status='timeout'))
        Path('/kaggle/working/probe_status.json').write_text(json.dumps(summary, indent=2))
        print(summary[-1], flush=True)


if __name__ == '__main__':
    main()
