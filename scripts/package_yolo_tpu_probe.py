"""Embed the TPU probe and data converter into a Kaggle kernel."""
from pathlib import Path
import base64
import io
import json
import tarfile

root = Path(__file__).resolve().parents[1]
output = root/'kaggle/exp038_yolo_tpu_probe'
files = ['shared/__init__.py', 'shared/data_split.py',
         'experiments/exp_002_yolo_seg/src/prepare_yolo.py',
         'kaggle/exp038_yolo_tpu_probe/probe.py']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for name in files:
        destination = 'probe.py' if name.endswith('/probe.py') else name
        archive.add(root/name, arcname=destination)
payload = base64.b64encode(buffer.getvalue()).decode()
wrapper = '''import os,sys,base64,io,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/kaggle/working/code'); root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as archive: archive.extractall(root)
os.environ['PYTHONPATH']=str(root)
os.execv(sys.executable,[sys.executable,str(root/'probe.py')])
'''.replace('PAYLOAD', repr(payload))
(output/'kernel.py').write_text(wrapper, encoding='utf-8')
metadata = json.loads((output/'kernel-metadata.json').read_text())
metadata['code_file'] = 'kernel.py'
(output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(len(wrapper))
