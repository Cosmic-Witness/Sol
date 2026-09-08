"""Probe pretrained Mask2Former with a fixed-shape, XLA-native instance loss."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PJRT_DEVICE']='TPU'
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from transformers import Mask2FormerForUniversalSegmentation
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    xr.use_spmd()
    device=xm.xla_device()
    stages=[]
    def stage(label, sync=True):
        print(json.dumps(dict(event='stage_start',stage=label)),flush=True)
        start=time.perf_counter()
        if sync:
            xm.mark_step()
            xm.wait_device_ops()
        row=dict(stage=label,seconds=time.perf_counter()-start)
        stages.append(row)
        print(json.dumps(row),flush=True)
        Path(f'/kaggle/working/{args.model}_{args.size}_stages.json').write_text(json.dumps(stages,indent=2))
    name=f'facebook/mask2former-swin-{args.model}-coco-instance'
    model=Mask2FormerForUniversalSegmentation.from_pretrained(name,num_labels=1,ignore_mismatched_sizes=True)
    model.to(device).train()
    stage('model_transfer')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    image=torch.randn(1,3,args.size,args.size,device=device)
    # Fixed query assignment avoids the host-side Hungarian matcher and dynamic
    # target shapes that force repeated XLA compilations.
    target_size=args.size//4
    masks=torch.zeros(100,target_size,target_size)
    for i in range(23):
        y=3+i*(target_size-8)//23
        masks[i,y:y+2,5:target_size-5]=1
    masks=masks.to(device)
    labels=torch.ones(100,dtype=torch.long)
    labels[:23]=0
    labels=labels.to(device)
    class_weight=torch.tensor([1.0,0.1]).to(device)
    stage('inputs_ready')
    timing=[]; losses=[]
    for step in range(3):
        started=time.perf_counter()
        optimizer.zero_grad()
        stage(f'{step}_forward_enter',sync=False)
        result=model(pixel_values=image)
        stage(f'{step}_forward')
        class_logits=result.class_queries_logits[0]
        mask_logits=result.masks_queries_logits[0]
        if mask_logits.shape[-2:] != masks.shape[-2:]:
            target=F.interpolate(masks[:,None],size=mask_logits.shape[-2:],mode='nearest')[:,0]
        else:
            target=masks
        positive=mask_logits[:23]
        bce=F.binary_cross_entropy_with_logits(positive,target[:23])
        probability=positive.sigmoid()
        dice=1-((2*(probability*target[:23]).sum((1,2))+1)/
                (probability.sum((1,2))+target[:23].sum((1,2))+1)).mean()
        classification=F.cross_entropy(class_logits,labels,weight=class_weight)
        loss=classification+5*bce+5*dice
        stage(f'{step}_loss')
        loss.backward()
        stage(f'{step}_backward')
        optimizer.step()
        stage(f'{step}_optimizer')
        losses.append(float(loss.detach().cpu()))
        if not math.isfinite(losses[-1]):
            raise RuntimeError('nonfinite loss')
        timing.append(time.perf_counter()-started)
        print(json.dumps(dict(event='step_complete',step=step,loss=losses[-1],seconds=timing[-1])),flush=True)
    report=dict(status='complete',model=name,size=args.size,instances=23,
                devices=xr.global_runtime_device_count(),seconds=timing,losses=losses,
                torch=torch.__version__,loss='fixed query class+BCE+Dice',stages=stages,
                note='synthetic feasibility, not quality evidence')
    Path(f'/kaggle/working/{args.model}_{args.size}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--size',type=int)
    args=parser.parse_args()
    if args.model: worker(args);return
    subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.6'])
    summary=[]
    for model,size in [('tiny',256),('tiny',1024)]:
        with Path(f'/kaggle/working/{model}_{size}.log').open('w') as log:
            try:
                run=subprocess.run([sys.executable,__file__,'--model',model,'--size',str(size)],stdout=log,stderr=subprocess.STDOUT,timeout=900)
                summary.append(dict(model=model,size=size,returncode=run.returncode))
            except subprocess.TimeoutExpired:
                summary.append(dict(model=model,size=size,status='timeout'))
        Path('/kaggle/working/status.json').write_text(json.dumps(summary,indent=2))
        print(summary[-1],flush=True)
        if summary[-1].get('status') == 'timeout' or summary[-1].get('returncode',0) != 0:
            break


if __name__=='__main__': main()
