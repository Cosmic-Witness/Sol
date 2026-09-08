"""Prepare three CPU validation kernels for exp040 checkpoints."""
from pathlib import Path
import base64,io,json,tarfile
root=Path(__file__).resolve().parents[1]
files=['shared/__init__.py','shared/data_split.py','shared/utils.py',
       'experiments/exp_033_inference/src/sweep.py']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
    for name in files: archive.add(root/name,arcname=name)
payload=base64.b64encode(buf.getvalue()).decode()
for epoch in (10,25,50):
    output=root/f'kaggle/exp042_eval_{epoch}';output.mkdir(parents=True,exist_ok=True)
    script='''import os,sys,subprocess,base64,io,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as a:a.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools','pandas'])
base=list(Path('/kaggle/input').rglob('checkpoints/best.pt'));state=list(Path('/kaggle/input').rglob('epoch_EPOCH.pt'))
assert len(base)==len(state)==1,(base,state)
ann=next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'));images=ann.parent/'train_images'
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-u','-m','experiments.exp_033_inference.src.sweep','--weights',str(base[0]),'--state',str(state[0]),'--annotations',str(ann),'--images',str(images),'--output','/kaggle/working/results','--variants','identity'],cwd=root)
'''.replace('PAYLOAD',repr(payload)).replace('EPOCH',str(epoch))
    (output/'kernel.py').write_text(script,encoding='utf-8')
    meta=dict(id=f'cosmicwitness/sol-exp042-eval-{epoch}',title=f'Sol exp042 eval {epoch}',
              code_file='kernel.py',language='python',kernel_type='script',is_private=True,
              enable_gpu=False,enable_tpu=False,enable_internet=True,dataset_sources=[],
              competition_sources=['filament-segmentation-2026'],
              kernel_sources=['cosmicwitness/sol-exp002-yolo-seg','cosmicwitness/sol-exp040-yolo-tpu-train'],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(output)
