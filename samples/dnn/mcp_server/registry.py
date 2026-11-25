"""Reads models.yml: which models exist, which tool each belongs to, and where its files are."""

import os
import re
from pathlib import Path

import cv2 as cv
import yaml

HERE = Path(__file__).resolve().parent
TASKS = ("docRead", "detect", "segment")
ENGINES = ("ENGINE_OPENCV", "ENGINE_OPENVINO")
TARGETS = ("DNN_TARGET_CPU", "DNN_TARGET_OPENCL", "DNN_TARGET_OPENCL_FP16", "DNN_TARGET_NPU")

MODELS = {}


def load_registry(path=HERE / "models.yml"):
    """Read models.yml and check every entry, so that a typo stops the server at startup."""
    with open(path) as f:
        models = yaml.safe_load(f) or {}
    for name, spec in models.items():
        for key in ("task", "adapter", "engine", "target", "model"):
            if key not in spec:
                raise ValueError(f"models.yml: '{name}' has no '{key}'")
        if spec["task"] not in TASKS:
            raise ValueError(f"models.yml: '{name}' has task '{spec['task']}', expected one of {TASKS}")
        if spec["engine"] not in ENGINES:
            raise ValueError(f"models.yml: '{name}' has engine '{spec['engine']}', expected one of {ENGINES}")
        if spec["target"] not in TARGETS:
            raise ValueError(f"models.yml: '{name}' has target '{spec['target']}', expected one of {TARGETS}")
    MODELS.clear()
    MODELS.update(models)


def get_model(name, task):
    """The models.yml entry of a model, after checking that it belongs to this tool."""
    spec = MODELS.get(name)
    if spec is None:
        known = ", ".join(n for n, s in MODELS.items() if s["task"] == task)
        raise ValueError(f"unknown model '{name}'. Models for {task}: {known}")
    if spec["task"] != task:
        raise ValueError(f"'{name}' is a {spec['task']} model; use the {spec['task']} tool for it")
    return spec


def find_file(name):
    """Look a model file or folder up as given, then in OPENCV_DOWNLOAD_CACHE_DIR, like the
    other samples in samples/dnn do. Returns None when it is in neither place."""
    if os.path.exists(name):
        return os.path.abspath(name)
    cache = os.environ.get("OPENCV_DOWNLOAD_CACHE_DIR")
    if cache and os.path.exists(os.path.join(cache, name)):
        return os.path.join(cache, name)
    return None


def forced_engine():
    """Why the server and validate.py must not start, or None. OPENCV_FORCE_DNN_ENGINE replaces
    the engine that every readNet*() call asks for, so models.yml would be ignored, and the two
    runs that validate.py compares would come from the same engine."""
    value = os.environ.get("OPENCV_FORCE_DNN_ENGINE", "").strip()
    if value not in ("", "0"):
        return (f"OPENCV_FORCE_DNN_ENGINE={value} replaces the engine of every model; "
                "unset it, and choose engines in models.yml instead")
    return None


def engine_missing(engine):
    """Why this cv2 can't run an engine, or None when it can.

    Without OpenVINO, ENGINE_OPENVINO falls back to ENGINE_OPENCV with only a warning, and the
    results look normal. Checking the build here keeps such a model from running on the wrong engine.
    """
    if engine == "ENGINE_OPENVINO" and not re.search(r"OpenVINO:\s+YES", cv.getBuildInformation()):
        return "this cv2 was built without OpenVINO (WITH_OPENVINO=ON)"
    return None
