"""Mask2Former instance training with host matching and static TPU loss shapes."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, SwinConfig

from shared.data_split import make_split, assert_disjoint
from shared.query_mask_loss import match_queries, query_mask_loss


def load_data(annotations):
    coco = json.loads(annotations.read_text(encoding='utf-8'))
    split = make_split(str(annotations)); assert_disjoint(split)
    records = {row['id']: row for row in coco['images']}
    instances = defaultdict(list)
    for row in coco['annotations']:
        instances[row['image_id']].append(row)
    return split, records, instances


def sample(record, instances, image_dir, size, rng):
    image = cv2.imread(str(image_dir / record['file_name']), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (record['height'], record['width']):
        raise ValueError(f"Invalid photograph {record['file_name']}")
    target_size = size // 4
    target = np.zeros((32, target_size, target_size), dtype=np.float32)
    active = np.zeros(32, dtype=np.float32)
    if len(instances) > 32:
        raise ValueError('More than 32 targets; do not truncate')
    for i, annotation in enumerate(instances):
        seg = annotation['segmentation']
        if isinstance(seg, list):
            seg = mu.merge(mu.frPyObjects(seg, image.shape[0], image.shape[1]))
        elif isinstance(seg['counts'], list):
            seg = mu.frPyObjects(seg, image.shape[0], image.shape[1])
        mask = mu.decode(seg)
        # Area targets retain narrow instances that nearest-neighbor can erase.
        target[i] = cv2.resize(mask.astype(np.float32), (target_size, target_size),
                               interpolation=cv2.INTER_AREA)
        active[i] = float(np.any(mask))
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < .5:
        image = np.fliplr(image); target = np.flip(target, axis=2)
    if rng.random() < .5:
        image = np.flipud(image); target = np.flip(target, axis=1)
    pixels = np.repeat(image[None], 3, axis=0).astype(np.float32) / 255
    pixels -= np.array([.485, .456, .406], np.float32)[:, None, None]
    pixels /= np.array([.229, .224, .225], np.float32)[:, None, None]
    return (torch.from_numpy(np.ascontiguousarray(pixels))[None],
            torch.from_numpy(np.ascontiguousarray(target))[None],
            torch.from_numpy(active)[None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--backend', choices=['cpu', 'tpu'], default='tpu')
    parser.add_argument('--probe-steps', type=int, default=0)
    parser.add_argument('--smoke-model', action='store_true')
    parser.add_argument('--seconds', type=int, default=36000)
    parser.add_argument('--max-epochs', type=int, default=50)
    args = parser.parse_args()
    if args.smoke_model and not args.probe_steps:
        raise ValueError('The small random model is only a lifecycle check')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(2026); torch.set_num_threads(4); cv2.setNumThreads(1)
    rng = np.random.default_rng(2026)
    if args.backend == 'tpu':
        os.environ['PJRT_DEVICE'] = 'TPU'
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        xr.use_spmd(); device = xm.xla_device()
        def sync():
            xm.mark_step(); xm.wait_device_ops()
    else:
        device = torch.device('cpu')
        def sync():
            pass
    events = []
    def event(name, **values):
        row = dict(event=name, elapsed=time.perf_counter()-started, **values)
        events.append(row)
        with (args.output/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    annotations = args.annotations or next(Path('/kaggle/input').rglob(
        'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    images = args.images or next(Path('/kaggle/input').rglob('train_images'))
    split, records, instances = load_data(annotations)
    if len(split.train_image_ids) != 974 or len(split.val_stems) != 106:
        raise ValueError('Canonical split changed')
    model_name = 'facebook/mask2former-swin-tiny-coco-instance'
    if args.smoke_model:
        backbone = SwinConfig(embed_dim=32, depths=[1,1,1,1], num_heads=[1,2,4,8],
                              window_size=4, out_features=['stage1','stage2','stage3','stage4'])
        config = Mask2FormerConfig(backbone_config=backbone, num_labels=1,
                                   encoder_layers=1, decoder_layers=2, num_queries=32)
        model = Mask2FormerForUniversalSegmentation(config)
    else:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True)
    model.to(device).train(); sync()
    event('model_ready', backend=args.backend, size=args.size, split=split.summary(),
          loss='Hungarian class+BCE+Dice', smoke=args.smoke_model)
    backbone, decoder = [], []
    for name, parameter in model.named_parameters():
        (backbone if 'pixel_level_module.encoder' in name else decoder).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=backbone, lr=1e-5, base_lr=1e-5),
        dict(params=decoder, lr=1e-4, base_lr=1e-4)], weight_decay=1e-4)
    reserve = min(1200, max(60, args.seconds//10))
    total_steps = args.probe_steps or args.max_epochs * len(split.train_image_ids)
    timings, losses = [], []
    order = list(split.train_image_ids); rng.shuffle(order)
    # Probe crowded records and then a different photograph to test reuse.
    if args.probe_steps:
        order = sorted(order, key=lambda k: -len(instances[k]))
    def save(step):
        sync()
        temporary = args.output/'checkpoint.partial.pt'
        saved = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     config=model.config.to_dict(), steps=step, size=args.size,
                     model_name=model_name, loss='Hungarian class+BCE+Dice', split_seed=2026)
        torch.save(saved, temporary)
        temporary.replace(args.output/'checkpoint.pt')
        event('checkpoint_saved', steps=step)
    for step in range(total_steps):
        if step >= total_steps:
            break
        if not args.probe_steps and time.perf_counter()-started >= args.seconds-reserve:
            event('deadline_reached', step=step); break
        step_start = time.perf_counter()
        if step and step % len(order) == 0:
            rng.shuffle(order)
        image_id = order[step % len(order)]
        pixels, target, present = sample(records[image_id], instances[image_id], images, args.size, rng)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixels.to(device)); sync()
        if args.probe_steps: event('forward', step=step+1)
        assigned = match_queries(outputs.class_queries_logits, outputs.masks_queries_logits,
                                 target, present)
        assigned = tuple(value.to(device) for value in assigned); sync()
        loss = query_mask_loss(outputs.class_queries_logits, outputs.masks_queries_logits, *assigned)
        sync()
        loss.backward(); sync()
        if args.probe_steps: event('backward', step=step+1)
        optimizer.step(); sync()
        scalar = float(loss.detach().cpu())
        if not math.isfinite(scalar):
            raise RuntimeError('Nonfinite training loss')
        timings.append(time.perf_counter()-step_start); losses.append(scalar)
        if step == 4 and not args.probe_steps:
            remaining = args.seconds-reserve-(time.perf_counter()-started)
            seconds_per_step = max(timings[2:]) * 1.25
            total_steps = min(total_steps, step+1+max(1,int(remaining/seconds_per_step)))
            event('schedule_resolved', total_steps=total_steps, warm_seconds=seconds_per_step)
        warmup = min(5, total_steps-1)
        progress = max(0., (step+1-warmup)/max(1,total_steps-warmup))
        scale = .05+.95*.5*(1+math.cos(math.pi*min(1.,progress)))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*scale
        if args.probe_steps or (step+1)%20 == 0:
            event('step_complete', step=step+1, seconds=timings[-1], loss=scalar)
        if (step+1)%100 == 0: save(step+1)
    completed = len(losses)
    if not completed:
        raise RuntimeError('No optimizer steps completed')
    save(completed)
    restored = torch.load(args.output/'checkpoint.pt', map_location='cpu', weights_only=True)
    reloaded = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(restored['config']))
    reloaded.load_state_dict(restored['model'], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        # Held-out photograph at the deployment resolution verifies reload/inference.
        val_id = split.val_image_ids[0]
        val_pixels, _, _ = sample(records[val_id], instances[val_id], images, args.size,
                                  np.random.default_rng(2026))
        predicted = reloaded(pixel_values=val_pixels)
        if not torch.isfinite(predicted.masks_queries_logits).all():
            raise RuntimeError('Reloaded model produced nonfinite masks')
    report = dict(status='complete', steps=completed, size=args.size, backend=args.backend,
                  smoke_model=args.smoke_model, losses=losses, seconds=timings,
                  checkpoint_reload=True, validation_forward=True,
                  loss='Hungarian class+BCE+Dice', elapsed_seconds=time.perf_counter()-started,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'train_report.json').write_text(json.dumps(report,indent=2))
    event('complete', steps=completed)


if __name__ == '__main__':
    main()
