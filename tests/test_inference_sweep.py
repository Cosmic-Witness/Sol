import numpy as np
import pytest
from experiments.exp_033_inference.src.sweep import encode, pq_rle, transform, inverse_mask
from shared.utils import compute_pq


@pytest.mark.parametrize("seed", range(8))
def test_rle_matches_canonical_metric(seed):
    rng = np.random.default_rng(seed)
    truth = [(rng.random((16, 16)) > .7).astype(np.uint8) for _ in range(seed % 4)]
    predicted = [m.copy() for m in truth[:2]]
    if seed % 2:
        predicted.append((rng.random((16, 16)) > .7).astype(np.uint8))
    got = pq_rle([encode(m) for m in predicted], [encode(m) for m in truth])
    wanted = compute_pq(predicted, truth)
    for key in got:
        assert got[key] == pytest.approx(wanted[key])


@pytest.mark.parametrize("variant", ["identity", "shift_1", "shift_2", "shift_-2"])
def test_native_alignment_roundtrip(variant):
    mask = np.zeros((32, 32), np.uint8); mask[8:15, 9:19] = 1
    np.testing.assert_array_equal(inverse_mask(transform(mask, variant), variant), mask)


@pytest.mark.parametrize('grow', [-1, 0])
def test_threshold_sweep_matches_separate_painting(grow, monkeypatch):
    from experiments.exp_033_inference.src import sweep
    from shared.utils import paint_panoptic
    monkeypatch.setattr(sweep, 'paint_panoptic',
                        lambda candidates, min_area: paint_panoptic(
                            candidates, min_area, image_size=(32, 32)))
    masks = []
    for x, y in [(3, 3), (7, 7), (8, 8), (17, 17)]:
        mask = np.zeros((32, 32), np.uint8)
        mask[x:x+10, y:y+10] = 1
        masks.append(mask)
    cache = {'photo': [dict(score=s, rle=encode(m)) for s, m in
                       zip([.8, .5, .5, .2], masks)]}
    records = {'photo': [('a', [encode(masks[0]), encode(masks[3])]),
                         ('b', [encode(masks[1])])]}
    got = sweep.evaluate_thresholds(cache, records, [.1, .3, .5, .7], grow, 10)
    for row in got:
        expected, _ = sweep.evaluate(cache, records, row['conf'], grow, 10)
        assert row == expected


def test_sparse_votes_match_dense_votes_for_every_threshold():
    from experiments.exp_033_inference.src.cached_tta import vote_rle
    from pycocotools import mask as mu
    rng = np.random.default_rng(2026)
    masks = [(rng.random((32, 32)) > .6).astype(np.uint8) for _ in range(12)]
    rles = [encode(mask) for mask in masks]
    votes = np.sum(masks, axis=0)
    for needed in range(1, 13):
        np.testing.assert_array_equal(mu.decode(vote_rle(rles, needed)), votes >= needed)


def test_dihedral_transforms_preserve_color_channels():
    from experiments.exp_007_tta.src.dihedral import apply_transform, invert_transform
    image = np.arange(8*9*3, dtype=np.uint8).reshape(8, 9, 3)
    for mirror in (False, True):
        for rotations in range(4):
            transformed = apply_transform(image, rotations, mirror)
            assert transformed.ndim == 3 and transformed.shape[2] == 3
            np.testing.assert_array_equal(invert_transform(transformed, rotations, mirror), image)
