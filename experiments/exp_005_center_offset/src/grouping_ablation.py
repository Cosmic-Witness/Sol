"""Is center/offset grouping actually better than connected components?

Tested before spending TPU quota, the way the spine idea was — that one measured
+0.0005 and was abandoned. Ground-truth instances are rendered, the targets are
derived from them, noise is added to imitate an imperfect network, and both
decoders are scored against the truth.

The noise matters. A decoder handed perfect targets recovers perfectly and
proves nothing; what needs establishing is whether grouping still holds up when
the offsets are wrong by a realistic margin.
"""
import json
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, "/home/user/Sol")
from experiments.exp_005_center_offset.src.group import group, targets_from_labels
from experiments.exp_005_center_offset.src.prepare import render_instances
from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq

SIZE = 1024
ANN = ("/tmp/claude-0/-home-user-Sol/57440e45-b682-5d96-b04e-bbd74d8d24a6/"
       "scratchpad/ann/MAGFiLO_1.0_Annotations_kaggle2026_train.json")


def main() -> None:
    coco = json.load(open(ANN))
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    val_ids = make_split(ANN).val_image_ids[:50]
    rng = np.random.default_rng(0)

    print(f"{'offset noise':>13} {'CC PQ':>8} {'CC RQ':>7} {'C/O PQ':>8} {'C/O RQ':>7} {'delta':>8}")
    for noise in (0.0, 5.0, 15.0, 30.0):
        cc_rows, co_rows = [], []
        for image_id in val_ids:
            labels = render_instances(by_image[image_id], SIZE)
            n = int(labels.max())
            if n == 0:
                continue
            truth = [(labels == i).astype(np.uint8) for i in range(1, n + 1)]
            truth = [t for t in truth if t.sum() >= 150]
            if not truth:
                continue

            mask = (labels > 0).astype(np.uint8)
            heatmap, offsets = targets_from_labels(labels)
            if noise:
                offsets = offsets + rng.normal(0, noise, offsets.shape).astype(np.float32)

            n_cc, cc_labels = cv2.connectedComponents(mask, connectivity=8)
            cc = [m for m in ((cc_labels == i).astype(np.uint8) for i in range(1, n_cc))
                  if m.sum() >= 150]
            co = group(mask, heatmap, offsets)

            cc_rows.append(compute_pq(cc, truth))
            co_rows.append(compute_pq(co, truth))

        c, o = aggregate_pq(cc_rows), aggregate_pq(co_rows)
        print(f"{noise:>13.0f} {c['pq']:>8.4f} {c['rq']:>7.3f} "
              f"{o['pq']:>8.4f} {o['rq']:>7.3f} {o['pq'] - c['pq']:>+8.4f}")


if __name__ == "__main__":
    main()
