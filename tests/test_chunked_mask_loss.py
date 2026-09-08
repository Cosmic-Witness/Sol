"""Differential checks against the installed upstream criterion, including backward."""
from types import SimpleNamespace

import pytest
import torch
from ultralytics.utils.loss import v8SegmentationLoss

from shared.chunked_mask_loss import ChunkedSegmentationLoss


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def inputs(overlap, empty=False, anchors=83):
    torch.manual_seed(2026)
    b, channels, h, w = 2, 4, 13, 17
    fg = torch.rand(b, anchors) > .15
    fg[1] = False  # one empty photograph, despite another having positives
    if empty:
        fg[:] = False
    idx = torch.randint(0, 3, (b, anchors))  # repeated assignments to each truth
    boxes = torch.tensor([1.2, 2.7, 13.6, 11.4]).repeat(b, anchors, 1)
    labels = torch.randint(0, 4, (b, h, w)).float()
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])
    masks = labels if overlap else torch.stack([labels[i] == j for i in range(b) for j in (1, 2, 3)]).float()
    proto = torch.randn(b, channels, h, w, requires_grad=True)
    coefficients = torch.randn(b, anchors, channels, requires_grad=True)
    return fg, masks, idx, boxes, batch_idx, proto, coefficients, torch.tensor([h, w])


def criterion(overlap, chunk_size):
    # Only the segmentation component is compared; the full-model integration
    # is exercised separately, without constructing a detector for every case.
    instance = object.__new__(ChunkedSegmentationLoss)
    instance.overlap, instance.chunk_size = overlap, chunk_size
    return instance


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 8, 64])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("anchors", [17, 83])
def test_upstream_loss_and_gradients(overlap, chunk_size, autocast, anchors):
    args = inputs(overlap, anchors=anchors)
    reference = SimpleNamespace(overlap=overlap, single_mask_loss=v8SegmentationLoss.single_mask_loss)
    # Fractional boxes exercise crop semantics; repeated target indices catch
    # accidental instance averaging instead of positive-anchor averaging.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = v8SegmentationLoss.calculate_segmentation_loss(reference, *args)
        actual = criterion(overlap, chunk_size).calculate_segmentation_loss(*args)
    expected_grads = torch.autograd.grad(expected, args[5:7], retain_graph=True)
    actual_grads = torch.autograd.grad(actual, args[5:7])
    tolerance = dict(rtol=.025, atol=.001) if autocast else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    for got, wanted in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, wanted, **tolerance)


def test_empty_batch_has_finite_zero_gradients():
    args = inputs(True, empty=True)
    result = criterion(True, 8).calculate_segmentation_loss(*args)
    assert result.item() == 0
    for grad in torch.autograd.grad(result, args[5:7]):
        assert torch.count_nonzero(grad) == 0


def test_checkpoint_does_not_retain_expanded_targets():
    args = inputs(True)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = criterion(True, 8).calculate_segmentation_loss(*args)
    # Checkpoint may retain the shared HxW label map and CxHxW prototypes,
    # but no KxHxW float target/logit/BCE tensors from the chunk.
    assert (8, 13, 17) not in saved_shapes
    result.backward()


def model_and_batch(batch_size=1, empty=False):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    model = SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    model.args = get_cfg(overrides={"mask_ratio": 1, "overlap_mask": True})
    masks = torch.zeros(batch_size, 64, 64)
    masks[:, 8:56, 8:56] = 1
    n = 0 if empty else batch_size
    batch = dict(img=torch.rand(batch_size, 3, 64, 64), masks=masks,
                 batch_idx=torch.arange(n), cls=torch.zeros(n, 1),
                 bboxes=torch.tensor([[.5, .5, .75, .75]]).repeat(n, 1))
    return model, batch


def test_full_detector_loss_and_parameter_gradients():
    model, batch = model_and_batch()
    # Float64 isolates mathematical parity from reduction-order roundoff
    # amplified by an untrained backbone. FP32/BF16 are tested at the loss level.
    model.double()
    batch["img"] = batch["img"].double()
    batch["bboxes"] = batch["bboxes"].double()
    predictions = model(batch["img"])
    expected = v8SegmentationLoss(model)(predictions, batch)[0]
    actual = ChunkedSegmentationLoss(model)(predictions, batch)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected_grad = torch.autograd.grad(expected.sum(), parameters, retain_graph=True, allow_unused=True)
    actual_grad = torch.autograd.grad(actual.sum(), parameters, allow_unused=True)
    for got, wanted in zip(actual_grad, expected_grad):
        if wanted is None:
            assert got is None
        else:
            torch.testing.assert_close(got, wanted, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("empty", [True, False])
def test_batch_two_full_detector_backward(empty):
    model, batch = model_and_batch(batch_size=2, empty=empty)
    model.criterion = ChunkedSegmentationLoss(model)
    result = model.loss(batch)[0].sum()
    result.backward()
    assert torch.isfinite(result)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_training_callback_ema_validation_and_checkpoint(tmp_path):
    import json
    from ultralytics.utils.torch_utils import ModelEMA
    from experiments.exp_032_chunked.src.train import install_criterion

    model, batch = model_and_batch()
    trainer = SimpleNamespace(model=model, args=model.args, ema=ModelEMA(model), save_dir=tmp_path)
    install_criterion(trainer, 3, {"mask_ratio": 1, "overlap_mask": True})
    report = json.loads((tmp_path / "resolved_experiment.json").read_text())
    assert report["chunk_size"] == 3
    assert isinstance(trainer.ema.ema.criterion, ChunkedSegmentationLoss)
    # Two steps materialize optimizer state, update EMA, then validate its loss.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        model.loss(batch)[0].sum().backward()
        optimizer.step()
        trainer.ema.update(model)
    with torch.no_grad():
        assert torch.isfinite(trainer.ema.ema.loss(batch)[0]).all()
    # A criterion on a saved EMA must survive loading and inference.
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(trainer.ema.ema, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    with torch.no_grad():
        assert torch.isfinite(restored.loss(batch)[0]).all()
    trainer.args.mask_ratio = 2
    with pytest.raises(RuntimeError, match="configuration changed"):
        install_criterion(trainer, 3, {"mask_ratio": 1})
