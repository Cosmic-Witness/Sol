"""Does spine seeding actually recover fragmented filaments? Measured, not assumed.

Ground-truth masks are degraded to imitate the failure exp_001 exhibits — thin
necks eroded through, so single filaments break into pieces — and the two
decompositions are scored against the true instances.
"""
import sys, json
sys.path.insert(0, "/home/user/Sol")
import numpy as np, cv2
from collections import defaultdict
from experiments.exp_004_spine_tpu.src.decompose import decompose, connected_components
from shared.utils import compute_pq, aggregate_pq
from shared.data_split import make_split

SIZE, NATIVE = 1024, 2048
ANN = "/tmp/claude-0/-home-user-Sol/57440e45-b682-5d96-b04e-bbd74d8d24a6/scratchpad/ann/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

coco = json.load(open(ANN))
by_image = defaultdict(list)
for a in coco["annotations"]:
    by_image[a["image_id"]].append(a)

val_ids = make_split(ANN).val_image_ids[:60]
scale = SIZE / NATIVE
rng = np.random.default_rng(0)

def render(anns):
    """True per-instance masks, plus the union mask and spine map."""
    inst, spine = [], np.zeros((SIZE, SIZE), np.uint8)
    for a in anns:
        m = np.zeros((SIZE, SIZE), np.uint8)
        for ring in a.get("segmentation") or []:
            if len(ring) >= 6:
                cv2.fillPoly(m, [(np.asarray(ring, np.float32).reshape(-1,2)*scale).astype(np.int32)], 1)
        if m.sum() >= 150:
            inst.append(m)
            sp = a.get("spine")
            if sp and len(sp) >= 4:
                cv2.polylines(spine, [(np.asarray(sp, np.float32).reshape(-1,2)*scale).astype(np.int32)],
                              False, 1, 5)
    return inst, spine

def fragment(mask, k):
    """Erode then dilate: severs thin necks, leaving the body largely intact."""
    if k <= 0: return mask
    er = cv2.erode(mask, np.ones((k,k), np.uint8))
    return cv2.dilate(er, np.ones((k,k), np.uint8)) & mask

for k in (0, 3, 5, 7):
    cc_rows, sp_rows, frag = [], [], []
    for iid in val_ids:
        inst, spine = render(by_image[iid])
        if not inst: continue
        union = np.clip(np.sum(inst, axis=0), 0, 1).astype(np.uint8)
        degraded = fragment(union, k)
        cc = connected_components(degraded)
        sp = decompose(degraded, spine)
        frag.append(len(cc) / max(len(inst), 1))
        cc_rows.append(compute_pq(cc, inst))
        sp_rows.append(compute_pq(sp, inst))
    c, s = aggregate_pq(cc_rows), aggregate_pq(sp_rows)
    print(f"erode {k}: pred/true ratio {np.mean(frag):.2f} | "
          f"CC PQ {c['pq']:.4f} (RQ {c['rq']:.3f}) | "
          f"SPINE PQ {s['pq']:.4f} (RQ {s['rq']:.3f}) | delta {s['pq']-c['pq']:+.4f}")
