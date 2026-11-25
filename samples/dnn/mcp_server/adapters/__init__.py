"""The adapter of each model, by the name that models.yml gives in its "adapter" key."""

from .granite_docling import GraniteDocling
from .insid3 import Insid3
from .owlv2 import OwlV2
from .paddleocr_vl import PaddleOcrVl
from .sam3 import Sam3
from .yoloe26 import YoloE26

ADAPTERS = {
    "granite_docling": GraniteDocling,
    "paddleocr_vl": PaddleOcrVl,
    "yoloe26": YoloE26,
    "owlv2": OwlV2,
    "sam3": Sam3,
    "insid3": Insid3,
}
