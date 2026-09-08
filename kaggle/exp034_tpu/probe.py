"""Static-shape TPU forward probe for the existing exp_002 checkpoint. GPU disabled."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PJRT_DEVICE"] = "TPU"
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def worker(size):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh
    torch.set_num_threads(4)
    xr.use_spmd()
    count = xr.global_runtime_device_count()
    device = xm.xla_device()
    mesh = Mesh(np.arange(count), (count,), ("data",))
    weights = list(Path('/kaggle/input').rglob('checkpoints/best.pt'))
    if len(weights) != 1: raise RuntimeError('Expected one exp002 checkpoint')
    picture = next(next(Path('/kaggle/input').rglob('train_images')).glob('*.jpeg'))
    image = cv2.resize(cv2.imread(str(picture)), (size, size))
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2,0,1))).float()[None]/255
    model = YOLO(weights[0]).model.float().eval()
    model.requires_grad_(False)
    model.fuse(verbose=False)
    model.model[-1].export = True
    model.model[-1].format = 'torchscript'
    with torch.no_grad():
        reference = model(tensor)
    model.to(device)
    batch = tensor.repeat(count,1,1,1).to(device)
    xs.mark_sharding(batch, mesh, ('data', None, None, None))
    timing = []
    for _ in range(3):
        start = time.perf_counter()
        with torch.no_grad():
            prediction, proto = model(batch)
        xm.mark_step()
        raw, masks = prediction.cpu(), proto.cpu()
        timing.append(time.perf_counter()-start)
    report = dict(status='complete', size=size, devices=count, torch=torch.__version__,
                  batch_seconds=timing, prediction_shape=list(raw.shape), prototype_shape=list(masks.shape),
                  prediction_max_abs=float((raw[:1]-reference[0]).abs().max()),
                  prototype_max_abs=float((masks[:1]-reference[1]).abs().max()),
                  finite=bool(torch.isfinite(raw).all() and torch.isfinite(masks).all()))
    Path(f'/kaggle/working/probe_{size}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--size',type=int)
    args=parser.parse_args()
    if args.size:
        worker(args.size); return
    subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0'])
    reports=[]
    for size in (1024,2048):
        with Path(f'/kaggle/working/probe_{size}.log').open('w') as stream:
            try:
                result=subprocess.run([sys.executable,__file__,'--size',str(size)],stdout=stream,stderr=subprocess.STDOUT,timeout=1000)
                reports.append(dict(size=size,returncode=result.returncode))
            except subprocess.TimeoutExpired:
                reports.append(dict(size=size,status='timeout'))
        Path('/kaggle/working/probe_status.json').write_text(json.dumps(reports,indent=2))
        print(reports[-1],flush=True)


if __name__=='__main__': main()
