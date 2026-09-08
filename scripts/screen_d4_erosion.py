"""Bounded existing-weight D4 erosion screen; preserve the validated control."""
import json
from pathlib import Path
import time

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for


def main():
    output = Path('data/exp049_d4_erosion'); output.mkdir(exist_ok=True)
    cache = json.loads(Path('data/exp033_cached_tta/candidates_0.5_0.5.json').read_text())
    records, split = records_for(Path('data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train')/
                                 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    if set(cache) != set(records):
        raise ValueError('Validation photographs do not match')
    started = time.perf_counter()
    results = []
    for grow in (-1, -2, -3):
        path = output/f'grow_{grow}.json'
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            rows = evaluate_thresholds(cache, records, [.35,.4,.45], grow=grow,min_area=300)
            path.write_text(json.dumps(rows,indent=2))
        if grow == -1:
            control = next(row for row in rows if row['conf'] == .4)
            if abs(control['pq']-.45707595553373687) > 1e-10:
                raise ValueError(f'D4 control did not reproduce: {control}')
        results.extend(rows)
        print(json.dumps(dict(grow=grow, best=max(rows,key=lambda r:r['pq']),
                              elapsed=time.perf_counter()-started)),flush=True)
    (output/'summary.json').write_text(json.dumps(dict(
        split=split.summary(),sweep=results,best=max(results,key=lambda r:r['pq'])),indent=2))


if __name__ == '__main__':
    main()
