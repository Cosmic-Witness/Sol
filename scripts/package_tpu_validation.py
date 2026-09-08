"""Embed TPU validation dependencies into its Kaggle kernel."""
from pathlib import Path
import base64,io,json,tarfile
root=Path(__file__).resolve().parents[1]; output=root/'kaggle/exp039_tpu_validation'
files=['shared/__init__.py','shared/data_split.py','shared/utils.py',
       'experiments/exp_033_inference/src/sweep.py','kaggle/exp039_tpu_validation/validate.py']
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for name in files:
        archive.add(root/name,arcname='validate.py' if name.endswith('/validate.py') else name)
payload=base64.b64encode(buffer.getvalue()).decode()
script='''import os,sys,subprocess,base64,io,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['PJRT_DEVICE']='TPU'
root=Path('/kaggle/working/code'); root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as archive: archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools','pandas'])
os.environ['PYTHONPATH']=str(root); os.execv(sys.executable,[sys.executable,str(root/'validate.py')])
'''.replace('PAYLOAD',repr(payload))
(output/'kernel.py').write_text(script,encoding='utf-8')
print(len(script))
