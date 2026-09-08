"""Time CPU backends on one real native image; no training or quantization."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    image = cv2.imread(args.image)
    assert image.shape == (2048, 2048, 3)
    x = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    model = YOLO(args.weights).model.float().eval().requires_grad_(False)
    model.fuse(verbose=False)
    model.model[-1].export = True
    model.model[-1].format = 'torchscript'
    times = []
    with torch.inference_mode():
        for _ in range(3):
            start = time.perf_counter()
            reference = model(x)
            times.append(time.perf_counter() - start)
    print('torch_seconds', times, flush=True)
    onnx_path = output / 'exp002.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, x, str(onnx_path), opset_version=17,
                          input_names=['images'], output_names=['prediction', 'prototype'],
                          dynamo=False)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    ort_times = []
    for _ in range(3):
        start = time.perf_counter()
        outputs = session.run(None, {'images': x.numpy()})
        ort_times.append(time.perf_counter() - start)
    report = dict(checkpoint_sha256=sha, torch_seconds=times, onnx_seconds=ort_times,
                  max_abs=[float(np.max(np.abs(a.numpy()-b))) for a, b in zip(reference, outputs)],
                  finite=all(bool(np.isfinite(a).all()) for a in outputs),
                  providers=session.get_providers(), torch=torch.__version__, onnxruntime=ort.__version__,
                  note='Single-image forward feasibility; full validation required before deployment.')
    assert hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() == sha
    (output / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
