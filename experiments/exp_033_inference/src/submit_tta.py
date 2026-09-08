"""Build and verify a submission from both CPU shards; does not upload it."""
import argparse
import csv
import json
from pathlib import Path
import cv2
from pycocotools import mask as mu
from shared.utils import paint_panoptic, check_no_overlap
from .cached_tta import fuse
from .sweep import BASELINE_PQ, encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', nargs=2, required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--test-images', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.validation).read_text())
    assert abs(report['baseline']['pq']-BASELINE_PQ) < 1e-6
    assert len(report['sweep']) == 42, 'Wait for the full planned sweep'
    rule = max(report['sweep'], key=lambda r:r['pq'])
    assert rule['pq'] > BASELINE_PQ
    cache = {}
    shard_ids = set()
    for directory in args.shards:
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['checkpoint_sha256'] == 'd9c6641edb1fec427212d0c35b008c923361e9fcf5004905a69c883714d93389'
        assert manifest['device'] == 'cpu' and manifest['floor_conf'] == .25
        assert manifest['imgsz'] == 2048 and manifest['nms_iou'] == .6
        assert manifest['max_det'] == 100 and manifest['retina_masks']
        assert all(g['min_iou'] >= .9 and g['mean_iou'] >= .98 and
                   g['max_score_difference'] < 1e-4 for g in manifest['gate'])
        shard_ids.add(manifest['shard'])
        shard = json.loads((directory/'test_views.json').read_text())
        assert len(shard) == 90 and not cache.keys() & shard.keys()
        cache.update(shard)
    assert shard_ids == {0, 1}
    assert set(cache) == {p.name for p in Path(args.test_images).glob('*.jpeg')}
    assert len(cache) == 180
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    counts = {}
    for name, entries in sorted(cache.items()):
        candidates = []
        for entry in fuse(entries, rule['link'], rule['vote']):
            if entry['score'] < rule['conf']:
                continue
            mask = mu.decode(entry['rle'])
            if rule['grow'] == -1:
                mask = cv2.erode(mask, kernel)
            else:
                assert rule['grow'] == 0
            candidates.append((entry['score'], mask))
        masks = paint_panoptic(candidates, min_area=rule['min_area'])
        counts[name] = len(masks)
        for i, (_, mask, _) in enumerate(masks, 1):
            rows.append(dict(filament_id=f'{Path(name).stem}_{i}', segmentation_rle=encode(mask)['counts']))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(rows)
    check_no_overlap(str(output))
    result = dict(rule=rule, rows=len(rows), photographs=len(cache), instance_counts=counts)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(rule=rule, rows=len(rows), photographs=len(cache))), flush=True)


if __name__ == '__main__':
    main()
