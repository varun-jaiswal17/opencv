"""Runs each model on ENGINE_OPENCV and on ENGINE_OPENVINO over the same test images and compares
the results. The two engines are not expected to agree exactly, only within these limits:

    text    character error rate at most 0.05
    boxes   matched one to one with IoU >= 0.9 and a score difference <= 0.02
    masks   matched one to one with IoU >= 0.95

    python validate.py --testset /path/to/testset
    python validate.py --testset /path/to/testset --models yoloe26,owlv2

The test set has a folder per tool, docRead/, detect/ and segment/, holding that tool's images.
A file named in test_inputs in models.yml, such as insid3's reference image, is relative to it.
Exits with 1 when a comparison fails.

Refuses to start while OPENCV_FORCE_DNN_ENGINE is set, because that variable replaces the engine
that each run asks for, and both runs would come from one engine.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import registry
import runner

ENGINES = ("ENGINE_OPENCV", "ENGINE_OPENVINO")
MAX_CER = 0.05
MIN_BOX_IOU, MAX_SCORE_DIFF = 0.9, 0.02
MIN_MASK_IOU = 0.95


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def compare_text(a, b):
    cer = edit_distance(a.raw, b.raw) / max(len(a.raw), 1)
    return cer <= MAX_CER, f"character error rate {cer:.3f}, {len(a.raw)} and {len(b.raw)} characters"


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - w * h
    return w * h / union if union > 0 else 0.0


def mask_iou(a, b):
    a, b = a > 0, b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def match(a, b, iou, threshold):
    """Pair the results of two engines, best overlap first. A result without a partner fails,
    unless its score is so close to the threshold that the other engine may have dropped it."""
    pairs, problems = [], []
    unmatched = list(b)
    for x in sorted(a, key=lambda r: -(r.score or 0)):
        candidates = [y for y in unmatched if y.label == x.label]
        if candidates:
            y = max(candidates, key=lambda y: iou(x, y))
            if iou(x, y) > 0:
                unmatched.remove(y)
                pairs.append((x, y))
                continue
        if (x.score or 1) > threshold + MAX_SCORE_DIFF:
            problems.append(f"{x.label} {x.score:.2f} only on {ENGINES[0]}")
    for y in unmatched:
        if (y.score or 1) > threshold + MAX_SCORE_DIFF:
            problems.append(f"{y.label} {y.score:.2f} only on {ENGINES[1]}")
    return pairs, problems


def compare_boxes(a, b, threshold):
    pairs, problems = match(a, b, lambda x, y: box_iou(x.box, y.box), threshold)
    ious = [box_iou(x.box, y.box) for x, y in pairs]
    diffs = [abs(x.score - y.score) for x, y in pairs]
    problems += [f"{x.label} IoU {i:.2f}" for (x, _), i in zip(pairs, ious) if i < MIN_BOX_IOU]
    problems += [f"{x.label} score {x.score:.3f} vs {y.score:.3f}"
                 for (x, y), d in zip(pairs, diffs) if d > MAX_SCORE_DIFF]
    detail = f"{len(pairs)} boxes matched"
    if pairs:
        detail += f", lowest IoU {min(ious):.3f}, largest score difference {max(diffs):.4f}"
    return not problems, "; ".join([detail] + problems)


def compare_masks(a, b, threshold):
    pairs, problems = match(a, b, lambda x, y: mask_iou(x.mask, y.mask), threshold)
    ious = [mask_iou(x.mask, y.mask) for x, y in pairs]
    problems += [f"mask {x.label or ''} IoU {i:.3f}" for (x, _), i in zip(pairs, ious) if i < MIN_MASK_IOU]
    detail = f"{len(pairs)} masks matched" + (f", lowest IoU {min(ious):.3f}" if pairs else "")
    return not problems, "; ".join([detail] + problems)


def short_error(e):
    """The useful line of an exception; OpenCV's put the message after a header line, behind '> '."""
    lines = [line.lstrip("> ").strip() for line in str(e).splitlines()]
    lines = [line for line in lines if line and not line.startswith("OpenCV(")]
    return " ".join(lines)[:200] if lines else type(e).__name__


def compare(task, a, b, threshold):
    if task == "docRead":
        return compare_text(a, b)
    if task == "detect":
        return compare_boxes(a, b, threshold)
    return compare_masks(a, b, threshold)


def validate(name, testset):
    """One (model, image, result, detail) row per test image."""
    spec = registry.MODELS[name]
    task = spec["task"]
    images = sorted(p for p in (testset / task).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        return [(name, "-", "SKIP", f"no images in {testset / task}")]
    for engine in ENGINES:
        reason = runner.missing(name, engine)
        if reason:
            return [(name, "-", "SKIP", f"{engine}: {reason}")]

    inputs = dict(spec.get("test_inputs") or {})
    for key in ("reference_image", "reference_mask"):
        if key in inputs:
            inputs[key] = str(testset / inputs[key])
    threshold = inputs.get("score_threshold", spec.get("score_threshold", 0))

    rows = []
    for image in images:
        out, times = {}, []
        try:
            for engine in ENGINES:
                start = time.perf_counter()
                out[engine] = runner.run_model(name, task, str(image), engine=engine, **inputs)
                times.append(time.perf_counter() - start)
        except Exception as e:
            rows.append((name, image.name, "FAIL", f"{engine}: {short_error(e)}"))
            continue
        ok, detail = compare(task, out[ENGINES[0]], out[ENGINES[1]], threshold)
        detail += f" ({times[0]:.2f}s vs {times[1]:.2f}s)"
        rows.append((name, image.name, "PASS" if ok else "FAIL", detail))
    runner.unload(name)   # both engines' networks; the next model may be large
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = os.path.join(os.environ.get("OPENCV_TEST_DATA_PATH", ""), "dnn", "mcp")
    parser.add_argument("--testset", type=Path, default=Path(default),
                        help="default: $OPENCV_TEST_DATA_PATH/dnn/mcp")
    parser.add_argument("--models", help="comma-separated names from models.yml; default: all")
    args = parser.parse_args()
    reason = registry.forced_engine()
    if reason:
        sys.exit(reason)

    registry.load_registry()
    names = args.models.split(",") if args.models else list(registry.MODELS)
    unknown = [n for n in names if n not in registry.MODELS]
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)}")

    print(f"{'model':<14}{'image':<18}{'result':<8}detail  ({ENGINES[0]} vs {ENGINES[1]})")
    failed = False
    for name in names:
        for row in validate(name, args.testset):
            print(f"{row[0]:<14}{row[1]:<18}{row[2]:<8}{row[3]}", flush=True)
            failed |= row[2] == "FAIL"
    sys.exit(1 if failed else 0)
