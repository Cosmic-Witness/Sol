"""Photograph-cluster bootstrap for a fixed, already-selected TTA rule."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import evaluate, records_for


def pq(rows):
    tp, fp, fn, iou = np.asarray(rows).reshape(-1, 4).sum(axis=0)
    return iou/(tp+.5*fp+.5*fn) if tp+.5*fp+.5*fn else 0.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True); parser.add_argument('--views',required=True)
    parser.add_argument('--annotations',required=True); parser.add_argument('--summary',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    records,_=records_for(args.annotations)
    raw=json.loads(Path(args.baseline).read_text())
    baseline={n:[dict(score=e['features']['score'],rle=dict(size=e['size'],counts=e['counts'])) for e in x]
              for n,x in raw.items()}
    report=json.loads(Path(args.summary).read_text())
    rule=max(report['sweep'],key=lambda r:r['pq'])
    views=json.loads(Path(args.views).read_text())
    candidate={n:fuse(x,rule['link'],rule['vote']) for n,x in views.items()}
    _, before=evaluate(baseline,records)
    _, after=evaluate(candidate,records,rule['conf'],rule['grow'],rule['min_area'])
    clustered=defaultdict(lambda:[np.zeros(4),np.zeros(4)])
    for table,side in [(before,0),(after,1)]:
        for row in table.values():
            clustered[row['photograph']][side] += [row['tp'],row['fp'],row['fn'],row['iou_sum']]
    pairs=np.array(list(clustered.values()))
    rng=np.random.default_rng(2026); deltas=[]
    for _ in range(20000):
        sample=pairs[rng.integers(0,len(pairs),len(pairs))].sum(axis=0)
        deltas.append(pq(sample[1])-pq(sample[0]))
    result=dict(rule=rule,photographs=len(pairs),bootstrap_seed=2026,resamples=len(deltas),
                delta_mean=float(np.mean(deltas)),delta_p025=float(np.quantile(deltas,.025)),
                delta_p975=float(np.quantile(deltas,.975)),probability_positive=float(np.mean(np.array(deltas)>0)))
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
