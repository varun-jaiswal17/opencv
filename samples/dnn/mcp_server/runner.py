"""Runs a model on an image: finds its adapter, loads it once, and keeps it for the next call.

Nothing here knows about MCP, so validate.py and the tests call run_model() directly.
"""

import threading

import cv2 as cv

import registry
from adapters import ADAPTERS

_adapters = {}               # (name, engine, target) -> loaded adapter
_lock = threading.Lock()


def missing(name, engine=None):
    """Why a model can't run with this cv2 and these files, or None when it can."""
    spec = registry.MODELS[name]
    engine = engine or spec["engine"]
    cls = ADAPTERS.get(spec["adapter"])
    if cls is None:
        return f"no adapter called '{spec['adapter']}'"
    for key in cls.files:
        if key not in spec:
            return f"models.yml gives no '{key}'"
        if registry.find_file(spec[key]) is None:
            return f"'{spec[key]}' not found (looked in the current folder and OPENCV_DOWNLOAD_CACHE_DIR)"
    return registry.engine_missing(engine) or cls.missing(spec, engine)


def get_adapter(name, task, engine=None, target=None):
    """The adapter of a model, created on the first call. engine and target override models.yml."""
    spec = registry.get_model(name, task)
    engine = engine or spec["engine"]
    target = target or spec["target"]
    key = (name, engine, target)
    with _lock:
        if key not in _adapters:
            reason = missing(name, engine)
            if reason:
                raise RuntimeError(f"model '{name}' can't run on {engine}: {reason}")
            _adapters[key] = ADAPTERS[spec["adapter"]](spec, engine, target)
        return _adapters[key]


def run_model(name, task, image_path, engine=None, target=None, **inputs):
    adapter = get_adapter(name, task, engine, target)
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"can't read {image_path} as an image")
    with adapter.lock:       # loading and forward() of one model run one call at a time
        adapter.ensure_loaded()
        return adapter.run(image, **inputs)


def unload(name):
    """Drop the loaded networks of a model, to free memory."""
    with _lock:
        for key in [k for k in _adapters if k[0] == name]:
            del _adapters[key]


def list_models():
    """The models of each tool, and whether they can run here."""
    tools = {task: [] for task in registry.TASKS}
    for name, spec in registry.MODELS.items():
        item = {"name": name, "about": spec.get("about", ""), "engine": spec["engine"]}
        reason = missing(name)
        item["available"] = reason is None
        if reason:
            item["reason"] = reason
        tools[spec["task"]].append(item)
    return tools
