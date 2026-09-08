"""CPU-only D4 test cache, with a historical validation-prediction gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from ultralytics import YOLO

from experiments.exp_007_tta.src.dihedral import predict_views


def serialise(entries):
    return [[score, dict(size=rle['size'], counts=rle['counts'].decode('ascii'))]
            for score, rle in entries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--historical-views', required=True)
    parser.add_argument('--train-images', required=True)
    parser.add_argument('--test-images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    assert sha == 'd9c6641edb1fec427212d0c35b008c923361e9fcf5004905a69c883714d93389'
    model = YOLO(args.weights)
    model.model.eval().requires_grad_(False)
    model.overrides['device'] = 'cpu'
    historical = json.loads(Path(args.historical_views).read_text())
    gates = []
    for name in sorted(historical)[:1]:
        with torch.inference_mode():
            actual = serialise(predict_views(model, Path(args.train_images)/name, 2048, .25, 100))
        expected = historical[name]
        assert len(actual) == len(expected), (name, len(actual), len(expected))
        matrix = mu.iou([e[1] for e in actual], [e[1] for e in expected], [0]*len(expected))
        differences = [abs(a[0]-b[0]) for a,b in zip(actual,expected)]
        diagonal = np.diag(matrix) if actual else np.ones(1)
        gate = dict(photograph=name, count=len(actual),
                    min_iou=float(diagonal.min()), mean_iou=float(diagonal.mean()),
                    max_score_difference=max(differences, default=0.))
        gates.append(gate)
        (output/'gate.json').write_text(json.dumps(gates, indent=2))
        # The historical job used a different Ultralytics patch release. Scores
        # and candidates must agree; a few boundary pixels may differ.
        assert gate['min_iou'] >= .9 and gate['mean_iou'] >= .98 and gate['max_score_difference'] < 1e-4, gate
        print('gate', gate, flush=True)
    pictures = sorted(Path(args.test_images).glob('*.jpeg'))
    assert len(pictures) == 180
    cache = {}
    started = time.perf_counter()
    for picture in pictures[args.shard::2]:
        with torch.inference_mode():
            cache[picture.name] = serialise(predict_views(model, picture, 2048, .25, 100))
        (output/'test_views.json').write_text(json.dumps(cache))
        print('test', args.shard, len(cache), '/90', flush=True)
    assert hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() == sha
    manifest = dict(checkpoint_sha256=sha, shard=args.shard, photographs=len(cache),
                    seconds=time.perf_counter()-started, device='cpu', imgsz=2048,
                    floor_conf=.25, nms_iou=.6, max_det=100, retina_masks=True,
                    torch=torch.__version__, gate=gates)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
