"""YOLOE-26 from Ultralytics, exported with a fixed list of classes.

Export:
    from ultralytics import YOLOE
    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(["person", "car", ...])     # the classes become part of the model
    model.export(format="onnx", end2end=False)

The output has a row of 4 box values, one score per class and 32 mask coefficients for each of
the 8400 anchors, and needs NMS. The default end-to-end export (end2end=True) gives 300 rows of
x1, y1, x2, y2, score, class instead and needs no NMS; this adapter reads both, but OpenVINO
2026.4 can't compile the end-to-end head (its GatherElements check rejects the shapes).
"""

import cv2 as cv
import numpy as np

from results import Detection
from .base import Adapter


class YoloE26(Adapter):
    def load(self):
        super().load()
        s = self.spec
        p = cv.dnn.Image2BlobParams()
        p.scalefactor = (s["scale"],) * 3
        p.mean = (s["mean"],) * 3
        p.size = (s["width"], s["height"])
        p.swapRB = s["rgb"]
        p.paddingmode = cv.dnn.DNN_PMODE_LETTERBOX
        p.borderValue = (114, 114, 114)
        self.params = p

    def run(self, image, classes=None, score_threshold=None):
        if classes:
            unknown = sorted(set(classes) - set(self.labels))
            if unknown:
                raise ValueError(f"{unknown} are not among the classes this YOLOE-26 export was made with; "
                                 "export it again with model.set_classes() to add them, or use owlv2")
        threshold = self.spec["score_threshold"] if score_threshold is None else score_threshold

        self.net.setInput(cv.dnn.blobFromImageWithParams(image, self.params))
        out = self.net.forward()[0]   # the first output; the second holds the mask prototypes

        nc = len(self.labels)
        per_anchor = out.shape[0] in (4 + nc, 4 + nc + 32)
        if per_anchor:
            # one row per anchor: cx, cy, w, h, a score per class, (32 mask coefficients)
            rows = out.T
            scores_all = rows[:, 4:4 + nc]
            class_ids = scores_all.argmax(axis=1)
            scores = scores_all[np.arange(len(rows)), class_ids]
            cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
        else:
            # end-to-end head: x1, y1, x2, y2, score, class, (32 mask coefficients)
            scores, class_ids = out[:, 4], out[:, 5].astype(int)
            boxes = np.stack([out[:, 0], out[:, 1], out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]], axis=1)

        keep = scores >= threshold
        if classes:
            keep &= np.isin(class_ids, [self.labels.index(c) for c in classes])
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if per_anchor:
            idx = cv.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), class_ids.tolist(),
                                         threshold, self.spec["nms_threshold"])
            boxes, scores, class_ids = boxes[idx], scores[idx], class_ids[idx]

        size = (image.shape[1], image.shape[0])
        found = []
        for (x, y, w, h), score, c in zip(boxes, scores, class_ids):
            # undo the letterbox: from the 640x640 input back to the original image
            box = self.params.blobRectToImageRect((int(round(x)), int(round(y)), int(round(w)), int(round(h))), size)
            found.append(Detection(box=tuple(box), label=self.labels[c], score=float(score)))
        return found
