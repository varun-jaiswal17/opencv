"""What every adapter shares: reading its files, loading its networks once, and one lock."""

import threading

import cv2 as cv

import registry


class Adapter:
    # The models.yml keys that name a file or folder this adapter reads.
    files = ("model",)

    def __init__(self, spec, engine, target):
        self.spec = spec
        self.engine = getattr(cv.dnn, engine)
        self.target = getattr(cv.dnn, target)
        self.paths = {key: registry.find_file(spec[key]) for key in self.files}
        self.labels = None
        if "labels" in spec:
            with open(registry.HERE / spec["labels"]) as f:
                self.labels = f.read().splitlines()
        self.lock = threading.Lock()   # one cv.dnn.Net must not run two forwards at once
        self.loaded = False

    @classmethod
    def missing(cls, spec, engine):
        """What else the model needs that this cv2 doesn't have, or None."""
        return None

    def ensure_loaded(self):
        """Load the networks on the first call. The runner holds self.lock around this."""
        if not self.loaded:
            self.load()
            self.loaded = True

    def load(self):
        self.net = self.read_net("model")

    def read_net(self, key):
        net = cv.dnn.readNetFromONNX(self.paths[key], engine=self.engine)
        # ENGINE_OPENCV warns about any target, even the CPU; ENGINE_OPENVINO picks AUTO without one
        if self.engine != cv.dnn.ENGINE_OPENCV or self.target != cv.dnn.DNN_TARGET_CPU:
            net.setPreferableTarget(self.target)
        return net

    def run(self, image, **inputs):
        raise NotImplementedError
