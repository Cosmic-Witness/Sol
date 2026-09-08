import pytest
import torch

from shared.query_mask_loss import match_queries, query_mask_loss


def example():
    generator = torch.Generator().manual_seed(2026)
    classes = torch.randn(1, 5, 2, generator=generator)
    masks = torch.randn(1, 5, 16, 16, generator=generator)
    targets = torch.zeros(1, 4, 16, 16)
    targets[0, 0, 2:6, 3:5] = 1
    targets[0, 1, 8:10, 6:14] = 1
    active = torch.tensor([[1., 1., 0., 0.]])
    return classes, masks, targets, active


def test_target_permutation_and_padding_are_ignored():
    c, m, t, a = example()
    assigned = match_queries(c, m, t, a)
    permutation = [2, 1, 3, 0]
    t[:, 2:] = 1  # Inactive padding must have no effect.
    shuffled = match_queries(c, m, t[:, permutation], a[:, permutation])
    for first, second in zip(assigned, shuffled):
        torch.testing.assert_close(first, second)
    assert assigned[2].sum() == 2
    assert assigned[1].shape == m.shape


def test_loss_and_gradients_are_query_permutation_invariant():
    c, m, t, a = example()
    c.requires_grad_(); m.requires_grad_()
    assigned = match_queries(c, m, t, a)
    loss = query_mask_loss(c, m, *assigned)
    loss.backward()
    permutation = [4, 2, 0, 3, 1]
    cp = c.detach()[:, permutation].requires_grad_()
    mp = m.detach()[:, permutation].requires_grad_()
    shuffled_loss = query_mask_loss(cp, mp, *match_queries(cp, mp, t, a))
    shuffled_loss.backward()
    torch.testing.assert_close(loss, shuffled_loss)
    torch.testing.assert_close(c.grad[:, permutation], cp.grad)
    torch.testing.assert_close(m.grad[:, permutation], mp.grad)
    assert torch.isfinite(m.grad).all() and torch.count_nonzero(m.grad) > 0


def test_empty_record_still_trains_no_object_class():
    c, m, t, a = example()
    c.requires_grad_(); m.requires_grad_(); a.zero_()
    assigned = match_queries(c, m, t, a)
    assert assigned[0].eq(1).all() and assigned[2].sum() == 0
    loss = query_mask_loss(c, m, *assigned)
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(c.grad) > 0
    assert torch.count_nonzero(m.grad) == 0


def test_reject_nonfinite_and_excess_targets():
    c, m, t, a = example()
    with pytest.raises(ValueError, match='More instances'):
        match_queries(c[:, :1], m[:, :1], t, a)
    m[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        match_queries(c, m, t, a)
