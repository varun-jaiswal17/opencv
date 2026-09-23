# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

__all__ = []

import cv2 as cv

_native_docRead = cv.docRead

_BLOCK_TYPE_NAMES = {0: "title", 1: "text", 2: "table", 3: "form", 4: "signature"}


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class _ConstResult:
    def __init__(self, value):
        self._value = value

    def __call__(self):
        return self._value


def _wrap(value):
    if isinstance(value, dict):
        return _AttrDict((k, _wrap(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _line_to_dict(line):
    words = [w.text for w in line.words] if line.words else line.text.split()
    return {"text": line.text, "words": words}


def _cell_to_dict(cell):
    return {"row": cell.row, "col": cell.col, "text": cell.text, "is_header": cell.is_header}


def _field_to_dict(field):
    return {"key": field.key, "value": field.value}


def _block_to_dict(block):
    btype = int(block.type)
    d = {"type": _BLOCK_TYPE_NAMES.get(btype, "text")}
    if block.lines:
        d["lines"] = [_line_to_dict(line) for line in block.lines]
    if btype == 2:  # BLOCK_TABLE
        d["rows"] = block.rows
        d["cols"] = block.cols
        d["cells"] = [_cell_to_dict(cell) for cell in block.cells]
    if btype == 3:  # BLOCK_FORM
        d["fields"] = [_field_to_dict(field) for field in block.fields]
    return d


def _page_to_dict(page):
    words = [w.text for w in page.getWords()]
    tables = [_block_to_dict(block) for block in page.getTables()]
    sentences = list(page.getSentences())
    return {"page_number": page.page_number, "width": page.width, "height": page.height,
            "blocks": [_block_to_dict(block) for block in page.blocks],
            "getWords": _ConstResult(words),
            "getTables": _ConstResult(tables),
            "getSentences": _ConstResult(sentences)}


def _result_to_dict(result):
    m = result.metadata
    return {
        "metadata": {"model": m.model, "engine": m.engine, "pages": m.pages,
                     "tokens_used": m.tokens_used, "inference_time_ms": m.inference_time_ms},
        "pages": [_page_to_dict(page) for page in result.pages],
    }


def docRead(input_path, model_type, model_dir, engine="opencv", device="cpu",
            api_key="", prompt="", max_new_tokens=512, raw=False):
    result = _native_docRead(input_path, model_type, model_dir, engine, device,
                              api_key, prompt, max_new_tokens, raw)
    return list(result.raw_pages) if raw else _wrap(_result_to_dict(result))


docRead.__doc__ = _native_docRead.__doc__
docRead.__module__ = cv.__name__
cv.docRead = docRead
