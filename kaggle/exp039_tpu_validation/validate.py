"""End-to-end canonical PQ check for TPU YOLO inference; GPU disabled."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PJRT_DEVICE']='TPU'
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from experiments.exp_033_inference.src.sweep import encode, evaluate, records_for, BASELINE_PQ


def main():
    xr.use_spmd(); count=xr.global_runtime_device_count(); device=xm.xla_device()
    assert count==8
    weights=list(Path('/kaggle/input').rglob('checkpoints/best.pt'))
    assert len(weights)==1
    annotations=next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    images=next(Path('/kaggle/input').rglob('train_images'))
    records,_=records_for(annotations); names=sorted(records)
    model=YOLO(weights[0]).model.float().eval().requires_grad_(False)
    model.fuse(verbose=False); model.model[-1].export=True; model.model[-1].format='torchscript'
    model.to(device)
    mesh=Mesh(np.arange(count),(count,),('data',))
    predictor=SegmentationPredictor(overrides=dict(conf=.05,iou=.60,max_det=100,
                                                    retina_masks=True,task='segment',verbose=False))
    predictor.model=SimpleNamespace(names=model.names,end2end=False)
    cache={}
    for start in range(0,len(names),count):
        batch_names=names[start:start+count]; valid=len(batch_names)
        batch_names += [batch_names[-1]]*(count-valid)
        originals=[cv2.imread(str(images/name)) for name in batch_names]
        cpu=torch.from_numpy(np.stack([np.ascontiguousarray(im[...,::-1].transpose(2,0,1))
                                       for im in originals])).float()/255
        tensor=cpu.to(device); xs.mark_sharding(tensor,mesh,('data',None,None,None))
        with torch.no_grad(): prediction,proto=model(tensor)
        xm.mark_step(); prediction,proto=prediction.cpu(),proto.cpu()
        predictor.batch=(batch_names,None,None)
        results=predictor.postprocess((prediction,proto),cpu,originals)
        for name,result in zip(batch_names[:valid],results[:valid]):
            entries=[]
            if result.masks is not None:
                for score,mask in zip(result.boxes.conf.numpy(),result.masks.data.numpy()):
                    mask=mask.astype(np.uint8)
                    if mask.sum()>=40: entries.append(dict(score=float(score),rle=encode(mask)))
            cache[name]=entries
        Path('/kaggle/working/tpu_candidates.json').write_text(json.dumps(cache))
        print(start+valid,'/',len(names),flush=True)
    sweep=[evaluate(cache,records,conf,grow)[0] for conf in (.25,.30,.35,.40,.45) for grow in (-1,0)]
    report=dict(status='complete',device='TPU',devices=count,checkpoint=str(weights[0]),
                baseline_reference=BASELINE_PQ,best=max(sweep,key=lambda r:r['pq']),sweep=sweep,
                photographs=len(cache),torch=torch.__version__)
    Path('/kaggle/working/summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
