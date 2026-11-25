"""OWLv2 (google/owlv2-base-patch16): finds objects named in free text.

One ONNX file takes the image and the tokenized class names together (inputs pixel_values,
input_ids and attention_mask) and scores each of its 3600 boxes against every name.
"""

import cv2 as cv
import numpy as np

from results import Detection
from .base import Adapter
from .clip_tokenizer import ClipTokenizer

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class OwlV2(Adapter):
    files = ("model", "tokenizer")

    def load(self):
        super().load()
        self.tokenizer = ClipTokenizer(self.paths["tokenizer"])

    def run(self, image, classes=None, score_threshold=None):
        if not classes:
            raise ValueError('owlv2 finds the objects you name: pass classes, such as ["car", "red umbrella"]')
        threshold = self.spec["score_threshold"] if score_threshold is None else score_threshold
        size = self.spec["width"]

        # As Owlv2Processor does: pad to a square with gray at the bottom and right, resize, normalize.
        h, w = image.shape[:2]
        side = max(h, w)
        square = np.full((side, side, 3), 0.5, dtype=np.float32)
        square[:h, :w] = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255
        square = cv.resize(square, (size, size), interpolation=cv.INTER_AREA if side > size else cv.INTER_LINEAR)
        pixels = ((square - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]

        ids, mask = self.tokenizer.encode_batch([f"a photo of a {c}" for c in classes], self.spec["text_length"])
        self.net.setInput(pixels, "pixel_values")
        self.net.setInput(ids, "input_ids")
        self.net.setInput(mask, "attention_mask")
        logits, boxes = self.net.forward(["logits", "pred_boxes"])

        scores_all = 1 / (1 + np.exp(-logits[0]))       # (3600, number of classes)
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]
        keep = scores >= threshold
        # boxes are cx, cy, w, h relative to the padded square
        cx, cy, bw, bh = (boxes[0][keep] * side).T
        rects = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
        scores, class_ids = scores[keep], class_ids[keep]
        idx = cv.dnn.NMSBoxesBatched(rects.tolist(), scores.tolist(), class_ids.tolist(),
                                     threshold, self.spec["nms_threshold"])

        found = []
        for i in idx:
            x, y, bw, bh = rects[i]
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            found.append(Detection(box=(x1, y1, x2 - x1, y2 - y1), label=classes[class_ids[i]], score=float(scores[i])))
        return found
