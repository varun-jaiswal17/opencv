"""Tests of the tools, the runner and the comparisons in validate.py. A fake adapter stands in for
the models, so no model files are needed, except for the tokenizer test, which is skipped without them.

    cd samples/dnn/mcp_server && python -m pytest tests
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np
import pytest
from fastmcp import Client

import adapters
import registry
import runner
import server
import validate
from adapters.base import Adapter
from adapters.clip_tokenizer import ClipTokenizer
from results import Detection, Document, Mask


class FakeAdapter(Adapter):
    files = ()

    def load(self):
        pass

    def run(self, image, **inputs):
        h, w = image.shape[:2]
        if self.spec["task"] == "detect":
            return [Detection(box=(1, 2, 3, 4), label="car", score=0.9)]
        mask = np.zeros((h, w), np.uint8)
        mask[:h // 2] = 255
        return [Mask(mask=mask, box=(0, 0, w, h // 2), score=0.8, label=inputs.get("prompt"))]


def fake_spec(task):
    return {"task": task, "adapter": "fake", "engine": "ENGINE_OPENCV", "target": "DNN_TARGET_CPU", "model": "-"}


@pytest.fixture(autouse=True)
def fake_models(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    cv.imwrite(str(uploads / "img.png"), np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(server, "ALLOWED", [uploads])
    monkeypatch.setattr(server, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(runner, "_adapters", {})
    monkeypatch.setitem(adapters.ADAPTERS, "fake", FakeAdapter)
    monkeypatch.setitem(registry.MODELS, "fake_detect", fake_spec("detect"))
    monkeypatch.setitem(registry.MODELS, "fake_segment", fake_spec("segment"))


def call(tool, args):
    async def go():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, args, raise_on_error=False)
    return asyncio.run(go())


def test_four_tools_with_descriptions():
    async def go():
        async with Client(server.mcp) as client:
            return await client.list_tools()
    tools = asyncio.run(go())
    assert {t.name for t in tools} == {"docRead", "detect", "segment", "list_models"}
    assert all(t.description for t in tools)


def test_detect_returns_boxes():
    r = call("detect", {"image_path": "img.png", "model": "fake_detect"})
    assert not r.is_error
    assert r.structured_content == {"detections": [{"box": [1, 2, 3, 4], "class": "car", "score": 0.9}]}


def test_segment_saves_masks():
    r = call("segment", {"image_path": "uploads/img.png", "model": "fake_segment", "prompt": "dog"})
    mask = r.structured_content["masks"][0]
    assert Path(mask["file"]).is_file()
    assert mask["area"] == 32 and mask["class"] == "dog"


def test_path_outside_the_allowed_folders_is_refused(tmp_path):
    cv.imwrite(str(tmp_path / "secret.png"), np.zeros((8, 8, 3), np.uint8))
    for path in ("../secret.png", str(tmp_path / "secret.png")):
        r = call("detect", {"image_path": path, "model": "fake_detect"})
        assert r.is_error and "allowed folders" in r.content[0].text


def test_allow_adds_a_folder(tmp_path, monkeypatch):
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    cv.imwrite(str(pictures / "cat.png"), np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(server, "ALLOWED", server.ALLOWED + [pictures])
    assert not call("detect", {"image_path": str(pictures / "cat.png"), "model": "fake_detect"}).is_error
    assert not call("detect", {"image_path": "cat.png", "model": "fake_detect"}).is_error


def test_annotate_returns_a_picture():
    r = call("segment", {"image_path": "img.png", "model": "fake_segment", "prompt": "dog", "annotate": True})
    assert [c.type for c in r.content] == ["text", "image"]
    assert r.content[1].mime_type == "image/png"
    assert r.structured_content["masks"][0]["class"] == "dog"
    assert [c.type for c in call("detect", {"image_path": "img.png", "model": "fake_detect"}).content] == ["text"]


def test_forced_engine_is_refused(monkeypatch):
    monkeypatch.delenv("OPENCV_FORCE_DNN_ENGINE", raising=False)
    assert registry.forced_engine() is None
    monkeypatch.setenv("OPENCV_FORCE_DNN_ENGINE", "3")
    assert "OPENCV_FORCE_DNN_ENGINE" in registry.forced_engine()


def test_model_of_another_tool_is_refused():
    r = call("detect", {"image_path": "img.png", "model": "fake_segment"})
    assert r.is_error and "segment tool" in r.content[0].text


def test_unknown_model_lists_the_known_ones():
    r = call("detect", {"image_path": "img.png", "model": "nope"})
    assert r.is_error and "fake_detect" in r.content[0].text


def test_list_models_says_why_a_model_is_missing(monkeypatch):
    spec = dict(fake_spec("detect"), adapter="yoloe26", model="no/such/file.onnx")
    monkeypatch.setitem(registry.MODELS, "missing_detect", spec)
    item = next(m for m in call("list_models", {}).structured_content["detect"] if m["name"] == "missing_detect")
    assert not item["available"] and "not found" in item["reason"]


def test_models_yml_names_existing_adapters():
    registry.load_registry()
    assert all(spec["adapter"] in adapters.ADAPTERS for spec in registry.MODELS.values())


def test_clip_tokenizer_matches_clip():
    path = registry.find_file("owlv2/tokenizer.json") or registry.find_file("sam3/tokenizer.json")
    if path is None:
        pytest.skip("no CLIP tokenizer.json in OPENCV_DOWNLOAD_CACHE_DIR")
    tok = ClipTokenizer(path)
    # token ids from transformers' CLIPTokenizer
    assert tok.encode("a photo of a cat") == [49406, 320, 1125, 539, 320, 2368, 49407]
    assert tok.encode("Hot-dog") == [49406, 2069, 268, 1929, 49407]
    ids, mask = tok.encode_batch(["cat"], 4)
    assert ids.tolist() == [[49406, 2368, 49407, 0]] and mask.tolist() == [[1, 1, 1, 0]]


def test_compare_boxes():
    a = [Detection((10, 10, 100, 100), "car", 0.80)]
    assert validate.compare_boxes(a, [Detection((11, 10, 100, 100), "car", 0.81)], 0.3)[0]
    assert not validate.compare_boxes(a, [Detection((40, 40, 100, 100), "car", 0.80)], 0.3)[0]
    assert not validate.compare_boxes(a, [], 0.3)[0]
    # a box this close to the threshold may be dropped by one engine
    assert validate.compare_boxes([Detection((0, 0, 5, 5), "car", 0.31)], [], 0.3)[0]


def test_compare_text():
    assert validate.compare_text(Document("", "a" * 100), Document("", "a" * 99 + "b"))[0]
    assert not validate.compare_text(Document("", "hello world"), Document("", "jello wurld"))[0]
