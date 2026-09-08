"""Generate D4 test candidates on TPU after exp039's PQ gate passes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['PJRT_DEVICE']='TPU'
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from experiments.exp_007_tta.src.dihedral import TRANSFORMS,apply_transform,invert_transform
from experiments.exp_033_inference.src.sweep import encode


def main():
    gate=list(Path('/kaggle/input').rglob('summary.json'))
    gate=[p for p in gate if 'exp039' in str(p).lower() or 'tpu-validation' in str(p).lower()]
    assert len(gate)==1,gate
    validation=json.loads(gate[0].read_text())
    assert validation['status']=='complete' and validation['best']['pq']>=.435,validation['best']
    xr.use_spmd(); count=xr.global_runtime_device_count(); device=xm.xla_device(); assert count==8
    weights=list(Path('/kaggle/input').rglob('checkpoints/best.pt')); assert len(weights)==1
    sha=hashlib.sha256(weights[0].read_bytes()).hexdigest()
    test=next(Path('/kaggle/input').rglob('test_images')); pictures=sorted(test.glob('*.jpeg')); assert len(pictures)==180
    model=YOLO(weights[0]).model.float().eval().requires_grad_(False)
    model.fuse(verbose=False); model.model[-1].export=True; model.model[-1].format='torchscript'; model.to(device)
    mesh=Mesh(np.arange(count),(count,),('data',))
    predictor=SegmentationPredictor(overrides=dict(conf=.25,iou=.60,max_det=100,retina_masks=True,task='segment',verbose=False))
    predictor.model=SimpleNamespace(names=model.names,end2end=False)
    cache={p.name:[] for p in pictures}
    for rotations,mirror in TRANSFORMS:
        for start in range(0,len(pictures),count):
            selected=pictures[start:start+count]; valid=len(selected); selected += [selected[-1]]*(count-valid)
            views=[]
            for path in selected:
                grey=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE); raw=np.repeat(grey[...,None],3,axis=2)
                views.append(apply_transform(raw,rotations,mirror))
            cpu=torch.from_numpy(np.stack([np.ascontiguousarray(v[...,::-1].transpose(2,0,1)) for v in views])).float()/255
            tensor=cpu.to(device); xs.mark_sharding(tensor,mesh,('data',None,None,None))
            with torch.no_grad(): prediction,proto=model(tensor)
            xm.mark_step(); prediction,proto=prediction.cpu(),proto.cpu()
            predictor.batch=([str(p) for p in selected],None,None)
            results=predictor.postprocess((prediction,proto),cpu,views)
            for path,result in zip(selected[:valid],results[:valid]):
                if result.masks is None: continue
                for score,mask in zip(result.boxes.conf.numpy(),result.masks.data.numpy()):
                    restored=invert_transform(mask.astype(np.uint8),rotations,mirror)
                    if restored.sum()>=40: cache[path.name].append([float(score),encode(restored)])
            Path('/kaggle/working/test_views.json').write_text(json.dumps(cache))
            print(rotations,mirror,start+valid,'/180',flush=True)
    manifest=dict(status='complete',checkpoint_sha256=sha,validation_gate=validation['best'],device='TPU',devices=count,
                  photographs=len(cache),views=8,imgsz=2048,floor_conf=.25,nms_iou=.6,max_det=100,retina_masks=True)
    Path('/kaggle/working/manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
