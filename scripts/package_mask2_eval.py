"""Package CPU validation/submission generation for the exp046 training output."""
import base64
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = ['shared/__init__.py','shared/data_split.py','shared/utils.py',
         'experiments/exp_033_inference/src/sweep.py',
         'experiments/exp_046_mask2/src/evaluate.py']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for name in files:
        archive.add(root/name,arcname=name)
payload = base64.b64encode(buffer.getvalue()).decode()
directory = root/'kaggle/exp048_mask2_eval';directory.mkdir(exist_ok=True)
wrapper = f'''import os,sys,base64,io,tarfile,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.6','pycocotools','pandas'])
os.environ['PYTHONPATH']=str(root)
checkpoints=list(Path('/kaggle/input').rglob('checkpoint.pt'))
assert len(checkpoints)==1,checkpoints
annotations=next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
images=next(Path('/kaggle/input').rglob('train_images'))
test_images=next(Path('/kaggle/input').rglob('test_images'))
subprocess.run([sys.executable,'-u','-m','experiments.exp_046_mask2.src.evaluate',
    '--checkpoint',str(checkpoints[0]),'--annotations',str(annotations),
    '--images',str(images),'--test-images',str(test_images),
    '--output','/kaggle/working/results'],cwd=root,check=True,timeout=14000)
'''
(directory/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = dict(id='cosmicwitness/sol-exp048-mask2-eval',title='Sol exp048 mask2 eval',
                code_file='kernel.py',language='python',kernel_type='script',is_private=True,
                enable_gpu=False,enable_tpu=False,enable_internet=True,dataset_sources=[],
                competition_sources=['filament-segmentation-2026'],
                kernel_sources=['cosmicwitness/sol-exp046-mask2-train'],model_sources=[])
(directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(directory)
