"""Granite-Docling 258M: reads a whole page into DocTags, which doctags.py turns into Markdown.

Runs through the vlm module (opencv/opencv#30146), which handles the image tiling, the three
ONNX files and the decoding loop. "model" is the export folder that cv.vlm.create() describes.
"""

import cv2 as cv

from doctags import doctags_to_markdown
from results import Document
from .base import Adapter


class GraniteDocling(Adapter):
    model_type = "VLM_MODEL_GRANITE_DOCLING"

    @classmethod
    def missing(cls, spec, engine):
        if not hasattr(cv, "vlm"):
            return "this cv2 has no vlm module (opencv/opencv#30146)"
        if engine != "ENGINE_OPENCV":
            return "cv2.vlm runs on ENGINE_OPENCV only, until ENGINE_OPENVINO has a KV cache"
        return None

    def load(self):
        self.model = cv.vlm.create(getattr(cv.vlm, self.model_type), self.paths["model"])
        if self.target != cv.dnn.DNN_TARGET_CPU:   # ENGINE_OPENCV warns about any target, even the CPU
            self.model.setPreferableTarget(self.target)

    def generate(self, image, prompt):
        self.model.reset()   # so that one page doesn't see the previous page's KV cache
        return self.model.infer(image, prompt, self.spec["max_new_tokens"])

    def run(self, image, prompt=""):
        raw = self.generate(image, prompt)
        return Document(markdown=doctags_to_markdown(raw, image), raw=raw)
