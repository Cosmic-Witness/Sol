"""Fine-tune exp002 with native-resolution mask supervision on TPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PJRT_DEVICE']='TPU'
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from ultralytics.models.yolo.segment import SegmentationTrainer
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

SIZE=2048
MASK_RATIO=1
BATCH=8
EPOCHS=50
LR=1e-4


def cpu_state(model):
    return {key:value.detach().cpu() for key,value in model.state_dict().items()}


def main():
    torch.manual_seed(2026); np.random.seed(2026)
    xr.use_spmd(); count=xr.global_runtime_device_count(); device=xm.xla_device()
    assert count==8 and BATCH%count==0
    weights=list(Path('/kaggle/input').rglob('checkpoints/best.pt')); assert len(weights)==1
    sha=hashlib.sha256(weights[0].read_bytes()).hexdigest()
    prepared=Path('/kaggle/working/yolo_data')
    trainer=SegmentationTrainer(overrides=dict(
        model=str(weights[0]),data=str(prepared/'data.yaml'),device='cpu',imgsz=SIZE,
        batch=BATCH,epochs=EPOCHS,mask_ratio=MASK_RATIO,overlap_mask=True,
        optimizer='AdamW',lr0=LR,workers=2,plots=False,cache=False,
        mosaic=0.,mixup=0.,copy_paste=0.,degrees=15.,hsv_h=0.,hsv_s=0.,hsv_v=.15,
        fliplr=.5,flipud=.5,project='/kaggle/working/setup',name='train'))
    trainer._setup_train()
    model=trainer.model.to(device).train(); model.criterion=model.init_criterion()
    optimiser=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,T_max=EPOCHS,eta_min=LR*.01)
    mesh=Mesh(np.arange(count),(count,1,1,1),('data','c','h','w'))
    output=Path('/kaggle/working/training'); output.mkdir(exist_ok=True)
    history=[]; started=time.time()
    for epoch in range(1,EPOCHS+1):
        running=torch.zeros(5,device=device); batches=0
        for batch in trainer.train_loader:
            batch=trainer.preprocess_batch(batch)
            if len(batch['img'])!=BATCH: continue
            for key,value in batch.items():
                if isinstance(value,torch.Tensor): batch[key]=value.to(device)
            xs.mark_sharding(batch['img'],mesh,('data','c','h','w'))
            optimiser.zero_grad(); loss,components=model(batch); loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimiser.step(); xm.mark_step(); running+=components.detach(); batches+=1
        scheduler.step(); values=[float(x) for x in (running/max(batches,1)).cpu()]
        row=dict(epoch=epoch,losses=values,batches=batches,lr=optimiser.param_groups[0]['lr'],
                 minutes=(time.time()-started)/60)
        history.append(row)
        manifest=dict(status='running',checkpoint_sha256=sha,size=SIZE,mask_ratio=MASK_RATIO,
                      batch=BATCH,epochs=EPOCHS,lr0=LR,devices=count,history=history,
                      torch=torch.__version__)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(row),flush=True)
        if epoch in (10,25,50): torch.save(dict(epoch=epoch,state=cpu_state(model)),output/f'epoch_{epoch}.pt')
    manifest['status']='complete'; (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
