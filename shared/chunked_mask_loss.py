"""Bound the YOLO mask-loss workspace without changing its training objective.

Positive *anchors*, not just instances, each expand a full-size target and logit
map upstream. Chunk before target expansion, then checkpoint the entire chunk:
chunking alone would retain all BCE intermediates until backward. The shared
prototype tensor is still full resolution; this does not bound backbone memory.

Objective/adapter derived from Ultralytics v8.4.0 utils/loss.py (AGPL-3.0):
https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/utils/loss.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ultralytics
from torch.utils.checkpoint import checkpoint
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import xyxy2xywh

if ultralytics.__version__ != "8.4.0":
    raise RuntimeError("ChunkedSegmentationLoss requires ultralytics==8.4.0")


class ChunkedSegmentationLoss(v8SegmentationLoss):
    """Drop-in criterion for the pinned YOLO11 trainer, with identical gains."""

    def __init__(self, model, chunk_size: int = 8):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        super().__init__(model)
        self.chunk_size = chunk_size

    def loss(self, preds, batch):
        # YOLO11 only. 8.4.0's generic wrapper mistakes a batch of two prototype
        # tensors for a (prototype, semantic) tuple and references absent semantic
        # tensors on empty batches. Handle the YOLO11 tensor contract explicitly.
        proto = preds["proto"]
        if not isinstance(proto, torch.Tensor) or proto.ndim != 4:
            raise ValueError("exp_032 supports YOLO11 tensor prototypes only")
        coefficients = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        (fg, indices, boxes, _, _), detection, _ = self.get_assigned_targets_and_loss(preds, batch)
        losses = torch.zeros(5, device=self.device)
        losses[0], losses[2], losses[3] = detection[0], detection[1], detection[2]
        if fg.any():
            masks = batch["masks"].to(self.device).float()
            if masks.shape[-2:] != proto.shape[-2:]:
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
            size = torch.tensor(preds["feats"][0].shape[2:], device=self.device,
                                dtype=coefficients.dtype) * self.stride[0]
            losses[1] = self.calculate_segmentation_loss(
                fg, masks, indices, boxes, batch["batch_idx"].view(-1, 1), proto, coefficients, size)
        else:
            losses[1] = proto.sum() * 0 + coefficients.sum() * 0
        losses[1] *= self.hyp.box
        return losses * proto.shape[0], losses.detach()

    def _chunk(self, coefficients, proto, masks, indices, boxes, areas, vector_crop):
        # These allocations must be INSIDE checkpoint, including target lookup.
        targets = ((masks == (indices + 1)[:, None, None]).float()
                   if self.overlap else masks[indices])
        if not vector_crop:
            return self.single_mask_loss(targets, coefficients, proto, boxes, areas)
        # Upstream switches from rounded CPU slicing to coordinate comparisons
        # at 50 masks. Preserve the ORIGINAL image's branch, not the chunk's.
        logits = torch.einsum("in,nhw->ihw", coefficients, proto)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        h, w = proto.shape[-2:]
        x = torch.arange(w, device=proto.device, dtype=boxes.dtype)[None, None, :]
        y = torch.arange(h, device=proto.device, dtype=boxes.dtype)[None, :, None]
        x1, y1, x2, y2 = boxes[:, :, None].chunk(4, 1)
        crop = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        return ((bce * crop).mean((1, 2)) / areas).sum()

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
        proto, pred_masks, imgsz,
    ):
        _, _, height, width = proto.shape
        normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        areas = xyxy2xywh(normalized)[..., 2:].prod(2)
        boxes = normalized * proto.new_tensor([width, height, width, height])
        # Connect both heads even when an image (or the whole batch) is empty.
        loss = proto.sum() * 0 + pred_masks.sum() * 0
        for i in range(len(fg_mask)):
            positive = fg_mask[i].nonzero(as_tuple=True)[0]
            image_masks = masks[i] if self.overlap else masks[batch_idx.view(-1) == i]
            for anchors in positive.split(self.chunk_size):
                if not len(anchors):
                    continue
                inputs = (
                    pred_masks[i, anchors], proto[i], image_masks,
                    target_gt_idx[i, anchors], boxes[i, anchors], areas[i, anchors],
                    proto.is_cuda or len(positive) >= 50,
                )
                if torch.is_grad_enabled():
                    loss = loss + checkpoint(self._chunk, *inputs, use_reentrant=False)
                else:
                    loss = loss + self._chunk(*inputs)
        return loss / fg_mask.sum().clamp_min(1)
