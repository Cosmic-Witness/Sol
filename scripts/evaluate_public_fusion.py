"""Score the public Mask R-CNN predictions and measured fusions with exp033 D4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def rle(value):
    return {"size": value["size"], "counts": value["counts"].encode("ascii")}


def morph(values, grow):
    result = []
    for value in values:
        mask = mu.decode(rle(value))
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow > 0:
            mask = cv2.dilate(mask, KERNEL, iterations=grow)
        if np.any(mask):
            result.append(encode(mask))
    return result


def painted(candidates, min_area=300):
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def score(predictions, records):
    rows = []
    for name, passes in records.items():
        predicted = predictions[name]
        for _, truth in passes:
            rows.append(pq_rle(predicted, truth))
    return aggregate_pq(rows)


def yolo_predictions(cache, conf=0.4, grow=-1):
    result = {}
    for name, entries in cache.items():
        candidates = []
        for entry in entries:
            if float(entry["score"]) < conf:
                continue
            mask = mu.decode(rle(entry["rle"]))
            if grow < 0:
                mask = cv2.erode(mask, KERNEL, iterations=-grow)
            candidates.append((float(entry["score"]), mask))
        result[name] = painted(candidates)
    return result


def ious(first, second):
    if not first or not second:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    return np.asarray(mu.iou(first, second, [0] * len(second)), dtype=np.float64)


def ordered_union(primary, secondary):
    candidates = [(2.0, mu.decode(value)) for value in primary]
    candidates += [(1.0, mu.decode(value)) for value in secondary]
    return painted(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    public_payload = json.loads(Path(args.public).read_text())
    cache = json.loads(Path(args.yolo).read_text())
    records, _ = records_for(args.annotations)
    expected = set(records)
    if set(cache) != expected:
        raise ValueError("YOLO cache does not match canonical validation photographs")
    results = []

    public_variants = {}
    for variant, values in public_payload["variants"].items():
        if set(values) != expected:
            raise ValueError(f"public {variant} predictions do not match validation split")
        for grow in (-1, 0, 1):
            predictions = {name: morph(values[name], grow) for name in expected}
            results.append({"name": f"public_{variant}_grow{grow}",
                            **score(predictions, records)})
            if variant == "candidate" and grow in (-1, 0):
                public_variants[grow] = predictions

    yolo = yolo_predictions(cache)
    results.append({"name": "yolo_d4_control", **score(yolo, records)})
    public = public_variants[0]
    for threshold in (0.05, 0.10, 0.20, 0.30, 0.50):
        yolo_first = {}
        public_first = {}
        consensus = {}
        for name in sorted(expected):
            matrix = ious(yolo[name], public[name])
            novel_public = [value for j, value in enumerate(public[name])
                            if not len(yolo[name]) or float(matrix[:, j].max()) <= threshold]
            supported_yolo = [value for i, value in enumerate(yolo[name])
                              if len(public[name]) and float(matrix[i].max()) > threshold]
            yolo_first[name] = ordered_union(yolo[name], novel_public)
            public_first[name] = ordered_union(public[name], yolo[name])
            consensus[name] = supported_yolo
        results.append({"name": f"yolo_plus_public_novel_iou{threshold}",
                        **score(yolo_first, records)})
        results.append({"name": f"public_then_yolo_iou{threshold}",
                        **score(public_first, records)})
        results.append({"name": f"yolo_consensus_iou{threshold}",
                        **score(consensus, records)})

    for operation in ("intersection", "union"):
        for threshold in (0.05, 0.10, 0.20, 0.30, 0.50):
            predictions = {}
            for name in sorted(expected):
                first, second = yolo[name], public[name]
                matrix = ious(first, second)
                pairs = [(float(matrix[i, j]), i, j)
                         for i in range(len(first)) for j in range(len(second))
                         if float(matrix[i, j]) > threshold]
                used_first, used_second, candidates = set(), set(), []
                for _, i, j in sorted(pairs, reverse=True):
                    if i in used_first or j in used_second:
                        continue
                    used_first.add(i); used_second.add(j)
                    a, b = mu.decode(first[i]).astype(bool), mu.decode(second[j]).astype(bool)
                    combined = (a & b) if operation == "intersection" else (a | b)
                    candidates.append((2.0, combined))
                candidates.extend((1.0, mu.decode(value)) for i, value in enumerate(first)
                                  if i not in used_first)
                predictions[name] = painted(candidates)
            results.append({"name": f"matched_{operation}_iou{threshold}",
                            **score(predictions, records)})

    results.sort(key=lambda row: -row["pq"])
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results[:12], indent=2))


if __name__ == "__main__":
    main()
