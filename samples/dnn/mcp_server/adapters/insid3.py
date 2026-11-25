"""INSID3: cuts out the object marked in an example image, using only a frozen DINOv3.

A port of models/insid3.py from https://github.com/visinf/INSID3 (Apache-2.0), without the
optional CRF refinement. The ONNX file is a DINOv3 ViT export that takes pixel_values of any
size and returns last_hidden_state: the class token, the register tokens and one token per
16x16 patch, e.g. onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX.

The steps, on patch features of the example image (reference) and the image to segment (target):
 1. Remove what the features of every image share because of patch position: the main
    directions of the features of a black image, found with an SVD.
 2. Average the reference features inside the mask into one prototype. Target patches that are
    similar to it, and whose nearest reference patch lies inside the mask, are candidates.
 3. Group the target patches into clusters, start from the candidate cluster most similar to
    the prototype, and add the clusters that are similar to both it and the prototype.
"""

import cv2 as cv
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from results import Mask
from .base import Adapter

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 16


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Insid3(Adapter):
    def load(self):
        super().load()
        size = self.spec["width"]
        self.grid = size // PATCH
        # 1. The positional directions, from a black image (all zeros before normalization).
        black = normalize(self.features(np.zeros((size, size, 3), np.uint8)))
        centered = (black - black.mean(axis=0)).T          # (channels, patches)
        u, _, _ = np.linalg.svd(centered, full_matrices=False)
        basis = u[:, :self.spec["svd_components"]]
        self.debias = np.eye(len(basis)) - basis @ basis.T

    def features(self, image):
        """One feature vector per patch, (grid * grid, channels)."""
        size = self.spec["width"]
        rgb = cv.cvtColor(cv.resize(image, (size, size), interpolation=cv.INTER_LINEAR), cv.COLOR_BGR2RGB)
        pixels = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]
        self.net.setInput(pixels, "pixel_values")
        tokens = self.net.forward("last_hidden_state")[0]
        return tokens[-self.grid * self.grid:]             # drop the class and register tokens

    def run(self, image, prompt=None, reference_image=None, reference_mask=None):
        if prompt:
            raise ValueError("insid3 takes an example instead of a prompt: pass reference_image and reference_mask")
        if not reference_image or not reference_mask:
            raise ValueError("insid3 needs reference_image and reference_mask: an example image with the object marked")
        ref = cv.imread(reference_image)
        ref_mask = cv.imread(reference_mask, cv.IMREAD_GRAYSCALE)
        if ref is None or ref_mask is None:
            raise ValueError("can't read reference_image or reference_mask as an image")

        g = self.grid
        tgt = normalize(self.features(image))
        tgt_deb = normalize(tgt @ self.debias)
        ref_deb = normalize(normalize(self.features(ref)) @ self.debias)

        # 2. The prototype and the candidate patches.
        inside = self.downsample_mask(ref_mask)
        proto = normalize(ref_deb[inside].mean(axis=0))
        sim = tgt_deb @ proto
        forward = sim > 0
        if not forward.any():
            forward = sim > np.quantile(sim, 0.9)
        backward = inside[(tgt_deb @ ref_deb.T).argmax(axis=1)]
        candidates = forward & backward
        if not candidates.any():
            return []

        # 3. Clusters of the target patches (average linkage on cosine distance), then the seed.
        distances = np.clip(1 - tgt @ tgt.T, 0, 2)
        condensed = distances[np.triu_indices(len(tgt), k=1)]
        labels = fcluster(linkage(condensed, method="average"), t=1 - self.spec["tau"], criterion="distance") - 1
        k = labels.max() + 1
        count = np.bincount(labels, minlength=k)
        protos_deb = normalize(np.stack([tgt_deb[labels == i].mean(axis=0) for i in range(k)]))
        protos = normalize(np.stack([tgt[labels == i].mean(axis=0) for i in range(k)]))

        matched = np.bincount(labels[candidates], minlength=k)
        ids = np.flatnonzero(matched)
        seed = ids[np.argmax(protos_deb[ids] @ proto)]
        area = matched / count
        area[seed] = 1.0
        cross = np.bincount(labels, weights=sim, minlength=k) / count
        intra = protos @ protos[seed]
        score = cross * intra * area
        keep = score[labels] > self.spec["merge_threshold"]

        h, w = image.shape[:2]
        small = keep.reshape(g, g).astype(np.float32)
        mask = (cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR) > 0.5).astype(np.uint8) * 255
        if not mask.any():
            return []
        return [Mask(mask=mask, box=tuple(cv.boundingRect(mask)), score=float(score[seed]))]

    def downsample_mask(self, mask):
        """The reference mask at patch resolution, as a flat bool array with at least one patch set."""
        g = self.grid
        small = cv.resize((mask > 0).astype(np.float32), (g, g), interpolation=cv.INTER_LINEAR) > 0.5
        if not small.any():
            small = cv.resize((mask > 0).astype(np.uint8), (g, g), interpolation=cv.INTER_NEAREST) > 0
        if not small.any():
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                raise ValueError("reference_mask is empty: mark the object with white (non-zero) pixels")
            small[int(ys.mean() * g / mask.shape[0]), int(xs.mean() * g / mask.shape[1])] = True
        return small.ravel()
