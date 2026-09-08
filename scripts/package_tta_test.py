"""Build two self-contained CPU kernels; publishing is a separate CLI command."""
from pathlib import Path
import base64
import io
import json
import tarfile

root = Path(__file__).resolve().parents[1]
files = ['shared/__init__.py', 'shared/utils.py', 'shared/data_split.py',
         'experiments/exp_007_tta/src/dihedral.py',
         'experiments/exp_033_inference/src/harvest_test.py']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for name in files:
        archive.add(root/name, arcname=name)
payload = base64.b64encode(buffer.getvalue()).decode()
for shard in (0, 1):
    output = root/f'kaggle/exp037_test_{shard}'
    output.mkdir(parents=True, exist_ok=True)
    script = '''import os, sys, subprocess, base64, io, tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/kaggle/working/code'); root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as archive: archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools','pandas'])
weights=list(Path('/kaggle/input').rglob('checkpoints/best.pt'))
views=list(Path('/kaggle/input').rglob('views.json'))
assert len(weights)==len(views)==1
train=next(Path('/kaggle/input').rglob('train_images'))
test=next(Path('/kaggle/input').rglob('test_images'))
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-u','-m','experiments.exp_033_inference.src.harvest_test',
'--weights',str(weights[0]),'--historical-views',str(views[0]),'--train-images',str(train),
'--test-images',str(test),'--output','/kaggle/working/results','--shard',SHARD],cwd=root)
'''.replace('PAYLOAD', repr(payload)).replace('SHARD', repr(str(shard)))
    (output/'kernel.py').write_text(script, encoding='utf-8')
    metadata = dict(id=f'cosmicwitness/sol-exp037-test-{shard}',
                    title=f'Sol exp037 test {shard}', code_file='kernel.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=False, enable_tpu=False, enable_internet=True,
                    dataset_sources=[], competition_sources=['filament-segmentation-2026'],
                    kernel_sources=['cosmicwitness/sol-exp002-yolo-seg', 'cosmicwitness/sol-exp007-tta'],
                    model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(output)
