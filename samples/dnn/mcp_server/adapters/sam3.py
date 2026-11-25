"""SAM 3: cuts out every object that matches a short text prompt, such as "dog".

Uses the three-file ONNX export from https://github.com/wkentaro/sam3-onnx
(Hugging Face: wkentaro/sam3-onnx-models-v0.7.0):
    sam3_image_encoder.onnx     image (uint8, 3x1008x1008 RGB) -> image features
    sam3_language_encoder.onnx  tokens (int64, 1x32) -> language_mask, language_features
    sam3_decoder.onnx           features + language + box prompts -> boxes, scores, masks
"""

import cv2 as cv
import numpy as np

from results import Mask
from .base import Adapter
from .clip_tokenizer import ClipTokenizer

SIZE = 1008
FEATURES = ["vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"]


class Sam3(Adapter):
    files = ("model", "language_model", "decoder_model", "tokenizer")

    def load(self):
        self.image_net = self.read_net("model")
        self.language_net = self.read_net("language_model")
        self.decoder_net = self.read_net("decoder_model")
        self.tokenizer = ClipTokenizer(self.paths["tokenizer"])

    def run(self, image, prompt=None, reference_image=None, reference_mask=None):
        if not prompt:
            raise ValueError('sam3 needs a prompt that describes what to cut out, such as "dog"')
        h, w = image.shape[:2]

        rgb = cv.cvtColor(cv.resize(image, (SIZE, SIZE), interpolation=cv.INTER_LINEAR), cv.COLOR_BGR2RGB)
        self.image_net.setInput(np.ascontiguousarray(rgb.transpose(2, 0, 1)), "image")
        features = self.image_net.forward(FEATURES)

        tokens, _ = self.tokenizer.encode_batch([prompt], self.spec["text_length"])
        self.language_net.setInput(tokens, "tokens")
        language_mask, language_features = self.language_net.forward(["text_attention_mask", "text_memory"])

        # No box prompts: every box slot is marked unused.
        n = self.spec["max_boxes"]
        inputs = dict(zip(FEATURES, features))
        inputs.update(language_mask=language_mask.astype(bool), language_features=language_features,
                      box_coords=np.zeros((n, 1, 4), np.float32), box_labels=np.ones((n, 1), np.int64),
                      box_masks=np.ones((1, n), bool))
        for name, value in inputs.items():
            self.decoder_net.setInput(value, name)
        boxes, scores, masks = self.decoder_net.forward(["boxes", "scores", "masks"])

        found = []
        for box, score, mask in zip(boxes, scores, masks):
            if score < self.spec["score_threshold"]:
                continue
            # masks are probabilities at 288x288; boxes are x1, y1, x2, y2 relative to the image
            full = (cv.resize(mask[0], (w, h), interpolation=cv.INTER_LINEAR) > 0.5).astype(np.uint8) * 255
            x1, y1, x2, y2 = box * [w, h, w, h]
            found.append(Mask(mask=full, box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                              score=float(score), label=prompt))
        return found
