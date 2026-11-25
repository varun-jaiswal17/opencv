"""PaddleOCR-VL 0.9B: reads text, tables, formulas or charts, through the vlm module like Granite-Docling.

The prompt picks the task: "OCR:" (the default), "Table Recognition:", "Formula Recognition:" or
"Chart Recognition:". The model already answers in plain text or Markdown.
"""

from results import Document
from .granite_docling import GraniteDocling


class PaddleOcrVl(GraniteDocling):
    model_type = "VLM_MODEL_PADDLEOCR_VL"

    def run(self, image, prompt=""):
        raw = self.generate(image, prompt)
        return Document(markdown=raw, raw=raw)
