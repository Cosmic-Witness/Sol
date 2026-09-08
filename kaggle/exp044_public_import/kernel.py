"""Verify and expose the exact output of the public, clean 0.70 handoff."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from pycocotools import mask as mu

EXPECTED='09c0eae5fdff29a91f8a740475ecaf36'
files=[p for p in Path('/kaggle/input').rglob('submission.csv') if p.is_file()]
matches=[p for p in files if hashlib.md5(p.read_bytes(),usedforsecurity=False).hexdigest()==EXPECTED]
if len(matches)!=1: raise RuntimeError(dict(found=[str(p) for p in files],matching=len(matches)))
source=matches[0]; frame=pd.read_csv(source)
assert list(frame.columns)==['filament_id','segmentation_rle'] and len(frame)==1299
assert frame.filament_id.is_unique and not frame.isna().any().any()
stems=frame.filament_id.str.rsplit('_',n=1).str[0]
test_dirs=[p for p in Path('/kaggle/input').rglob('test_images') if p.is_dir()]
assert len(test_dirs)==1
test_stems={p.stem for p in test_dirs[0].glob('*.jpeg')}; assert len(test_stems)==180
assert set(stems).issubset(test_stems)
for stem,group in frame.groupby(stems):
    rles=[dict(size=[2048,2048],counts=str(x).encode('ascii')) for x in group.segmentation_rle]
    assert all(mu.area(r)>0 for r in rles)
    assert int(mu.area(mu.merge(rles)).item())==sum(int(mu.area(r).item()) for r in rles),stem
output=Path('/kaggle/working/submission.csv'); shutil.copy2(source,output)
assert hashlib.md5(output.read_bytes(),usedforsecurity=False).hexdigest()==EXPECTED
report=dict(status='complete',source=str(source),submission_md5=EXPECTED,rows=len(frame),
            photographs=len(set(stems)),missing_photographs=len(test_stems-set(stems)),
            overlap_check='passed',training_executed=False,
            provenance='Public Apache-2.0 model output; competition-train-only per model card.')
Path('/kaggle/working/run_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
