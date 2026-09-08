"""Package the same Mask2Former trainer for a real-data probe and full training."""
import base64
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = ['shared/__init__.py', 'shared/data_split.py', 'shared/query_mask_loss.py',
         'kaggle/exp046_mask2_train/train.py']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for name in files:
        archive.add(root/name, arcname='train.py' if name.endswith('/train.py') else name)
payload = base64.b64encode(buffer.getvalue()).decode()
for slug, arguments, timeout in [
    ('exp046-mask2-train', [], 39600),
    ('exp047-mask2-real-probe', ['--probe-steps','3','--seconds','2700'], 3300),
]:
    directory = root/'kaggle'/slug.replace('-', '_')
    directory.mkdir(exist_ok=True)
    wrapper = f'''import os,sys,base64,io,tarfile,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PJRT_DEVICE']='TPU'
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.6','pycocotools','scipy'])
os.environ['PYTHONPATH']=str(root)
subprocess.run([sys.executable,'-u',str(root/'train.py')]+{arguments!r},cwd=root,check=True,timeout={timeout-120})
'''
    (directory/'kernel.py').write_text(wrapper,encoding='utf-8')
    metadata = dict(id='cosmicwitness/sol-'+slug,title='Sol '+slug.replace('-', ' '),
                    code_file='kernel.py',language='python',kernel_type='script',is_private=True,
                    enable_gpu=False,enable_tpu=True,machine_shape='Tpu1VmV38',enable_internet=True,
                    dataset_sources=[],competition_sources=['filament-segmentation-2026'],
                    kernel_sources=[],model_sources=[])
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(directory)
