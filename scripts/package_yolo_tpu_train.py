"""Embed exp040 and its canonical data converter."""
from pathlib import Path
import base64,io,tarfile
root=Path(__file__).resolve().parents[1]; output=root/'kaggle/exp040_yolo_tpu_train'
files=['shared/__init__.py','shared/data_split.py','experiments/exp_002_yolo_seg/src/prepare_yolo.py',
       'kaggle/exp040_yolo_tpu_train/train.py']
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for name in files: archive.add(root/name,arcname='train.py' if name.endswith('/train.py') else name)
payload=base64.b64encode(buffer.getvalue()).decode()
script='''import os,sys,subprocess,base64,io,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['PJRT_DEVICE']='TPU'
root=Path('/kaggle/working/code'); root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as archive: archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0'])
ann=next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
images=ann.parent/'train_images'; prepared=Path('/kaggle/working/yolo_data')
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','experiments.exp_002_yolo_seg.src.prepare_yolo','--annotations',str(ann),'--images',str(images),'--output',str(prepared),'--val-fraction','.15'])
os.execv(sys.executable,[sys.executable,str(root/'train.py')])
'''.replace('PAYLOAD',repr(payload))
(output/'kernel.py').write_text(script,encoding='utf-8'); print(len(script))
