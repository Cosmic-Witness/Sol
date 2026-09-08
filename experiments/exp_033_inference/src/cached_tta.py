"""Re-evaluate cached D4 predictions with the missing emission-threshold sweep."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu
from .sweep import encode, evaluate, evaluate_thresholds, records_for, BASELINE_PQ


def vote_rle(masks, needed):
    """Exact k-of-n voting using sparse RLE intersections and unions."""
    levels = [None] * needed
    for i, mask in enumerate(masks):
        for k in range(min(i + 1, needed), 1, -1):
            both = mu.merge([levels[k - 2], mask], intersect=True)
            levels[k - 1] = both if levels[k - 1] is None else mu.merge([levels[k - 1], both])
        levels[0] = mask if levels[0] is None else mu.merge([levels[0], mask])
    result = levels[-1]
    counts = result['counts']
    return dict(size=result['size'], counts=counts.decode('ascii') if isinstance(counts, bytes) else counts)


def fuse(entries, link, vote):
    order=sorted(range(len(entries)),key=lambda i:-entries[i][0])
    if not order: return []
    ious=mu.iou([e[1] for e in entries],[e[1] for e in entries],[0]*len(entries))
    used=set(); output=[]
    for seed in order:
        if seed in used: continue
        members=[seed];used.add(seed)
        for i in order:
            if i not in used and ious[seed,i]>=link:
                members.append(i);used.add(i)
        binary=vote_rle([entries[i][1] for i in members], max(1,round(vote*len(members))))
        if mu.area(binary)<40: continue
        score=float(np.mean([entries[i][0] for i in members]))
        output.append(dict(score=score*(.5+.5*len(members)/8),rle=binary))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True)
    parser.add_argument('--views',required=True);parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    cv2.setNumThreads(1)
    records,_=records_for(args.annotations)
    source=json.loads(Path(args.baseline).read_text())
    baseline={name:[dict(score=e['features']['score'],rle=dict(size=e['size'],counts=e['counts'])) for e in entries] for name,entries in source.items()}
    fixed,_=evaluate(baseline,records)
    if abs(fixed['pq']-BASELINE_PQ)>1e-6: raise RuntimeError(f'Baseline mismatch {fixed}')
    print('baseline',fixed,flush=True)
    views=json.loads(Path(args.views).read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary_path=out/'summary.json'
    results=json.loads(summary_path.read_text())['sweep'] if summary_path.exists() else []
    for link,vote in [(0.5,0.5),(0.5,0.7),(0.4,0.5)]:
        path=out/f'candidates_{link}_{vote}.json'
        if path.exists():
            cache=json.loads(path.read_text())
        else:
            cache={}
            for i,(name,entries) in enumerate(views.items()):
                cache[name]=fuse(entries,link,vote)
                if (i+1)%10==0: print('fuse',link,vote,i+1,flush=True)
            path.write_text(json.dumps(cache))
        for grow in (0,-1):
            done={r['conf'] for r in results if r['link']==link and r['vote']==vote and r['grow']==grow}
            if done.issuperset((.15,.20,.25,.30,.35,.40,.45)):
                continue
            for row in evaluate_thresholds(cache,records,(.15,.20,.25,.30,.35,.40,.45),grow):
                if row['conf'] in done: continue
                row.update(link=link,vote=vote);results.append(row)
                print(row,flush=True)
                (out/'summary.json').write_text(json.dumps(dict(baseline=fixed,sweep=results,best=max(results,key=lambda r:r['pq'])),indent=2))


if __name__=='__main__':main()
