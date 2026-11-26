# DNN Engine Selection {#api_dnn_engine_selection}

## Detailed Description

OpenCV 5 introduces a selectable inference backend for the DNN module, referred to as the *engine*. All engines share the same public API - `cv::dnn::readNet()`, `net.forward()`, and related functions - so switching between them requires changing at most a single argument.

The engine is specified at model-load time and cannot be changed after the model has been loaded, as each engine uses a different internal graph representation.

### ENGINE_AUTO

`ENGINE_AUTO` is the default value for the engine parameter on all `readNet*()` functions. It lets OpenCV pick the engine: it currently resolves to `ENGINE_OPENCV`. The resolution is intentionally an implementation detail and may change as more engines are added, so code that does not need a specific engine should leave the default in place.

Because `ENGINE_AUTO` is the default, existing code that does not pass an engine argument requires no modification.

### ENGINE_OPENCV

`ENGINE_OPENCV` is OpenCV's built-in DNN engine, a ground-up rewrite of the inference graph introduced in OpenCV 5. It is built around a typed operation graph with shape inference, constant folding, and operator fusion, covering approximately 75-80% of the ONNX operator specification. Models that failed to load under OpenCV 4.x due to dynamic shapes or unsupported operators will typically load and run correctly under this engine.

The engine performs automatic attention fusion: the `MatMul` → `Softmax` → `MatMul` subgraph common to transformer architectures is recognised and collapsed into a single fused operation at load time, with no changes required to the model or calling code.

`ENGINE_OPENCV` also introduces native support for Large Language Models and Vision-Language Models. Built-in tokenizers, attention layers, decoding blocks, and KV-caching allow models such as Qwen, Gemma, and PaliGemma to run end-to-end through the standard `Net` API, with no external runtime required.

OpenCV 5.1 introduces initial GPU support for `ENGINE_OPENCV` through a CUDA backend. This work is still in progress: layer coverage and performance are being extended, and layers without a CUDA implementation run on CPU. In OpenCV 5.0 the new engine (then `ENGINE_NEW`) was CPU-only, and CUDA inference required the classic engine, which has since been removed. The CUDA backend is built when OpenCV is configured with CUDA, cuBLAS and either cuDNN (`WITH_CUDNN`) or the cuDNN graph API (`WITH_CUDNNJIT`). To try it, call `net.setPreferableBackend(DNN_BACKEND_CUDA)` and `net.setPreferableTarget(DNN_TARGET_CUDA)` or `DNN_TARGET_CUDA_FP16`.

@note The Darknet, Caffe and TensorFlow parsers have been removed in OpenCV 5; ONNX is the recommended format. TFLite models are still supported and are executed via `ENGINE_OPENCV`.

### ENGINE_ORT

`ENGINE_ORT` routes inference through a bundled ONNX Runtime (ORT) wrapper. OpenCV uses its own ONNX parser to construct the ORT graph internally, so only the ORT library is required at runtime, not the standalone onnx package. It applies to ONNX models only.

`ENGINE_ORT` must be enabled at compile time:

```bash
# CPU only
cmake -DWITH_ONNXRUNTIME=ON ..

# With NVIDIA GPU execution providers
cmake -DWITH_ONNXRUNTIME=ON -DDOWNLOAD_ONNXRUNTIME_GPU=ON ..
```

ORT execution providers are supported, including CUDA for NVIDIA hardware. This makes `ENGINE_ORT` the recommended choice for GPU-accelerated inference while GPU support for `ENGINE_OPENCV` is still in development.

### ENGINE_OPENVINO

`ENGINE_OPENVINO` hands the whole ONNX model to the OpenVINO runtime, which reads, compiles and runs it. OpenCV only passes the inputs in and the outputs back. It applies to ONNX models given by file path; loading from a memory buffer is not supported.

`ENGINE_OPENVINO` must be enabled at compile time, with OpenVINO linked into the dnn module:

```bash
cmake -DWITH_OPENVINO=ON -DOpenVINO_DIR=<openvino>/runtime/cmake ..
```

The model is compiled on the first `forward()`, or by `finalizeNet()`. Without a target it is compiled for OpenVINO's `AUTO` device, and OpenVINO chooses the device. With its default settings, AUTO leaves out the NPU and GPUs that are not Intel GPUs, prefers a discrete Intel GPU, then an integrated one, then the CPU, and runs the first inferences on the CPU while the model is still compiling for the GPU.

`setPreferableTarget()` fixes the device instead:

| Target | OpenVINO device |
|---|---|
| none | AUTO |
| `DNN_TARGET_CPU` | CPU |
| `DNN_TARGET_OPENCL` | GPU |
| `DNN_TARGET_OPENCL_FP16` | GPU |
| `DNN_TARGET_NPU` | NPU |

OpenCV does not set a precision: each device runs at OpenVINO's default inference precision, which is FP16 on the GPU (for both GPU targets) and FP32 on most CPUs (BF16 on CPUs with AMX). With AUTO, the first results can therefore come from the CPU in FP32 and later ones from a GPU in FP16. "GPU" is OpenVINO's first GPU, which is the Intel integrated GPU when there is one. Other targets, and devices that OpenVINO does not find (or a first GPU that is not an Intel GPU), fall back to `DNN_TARGET_CPU` with a warning. Changing the target after the model was compiled compiles it again on the next `forward()` for the new device, without reading the ONNX file again. The backend set with `setPreferableBackend()` is ignored. When OpenVINO is built as a DNN plugin (`DNN_PLUGIN_LIST=openvino`), `DNN_BACKEND_INFERENCE_ENGINE` is available but `ENGINE_OPENVINO` is not, and models fall back to `ENGINE_OPENCV` with a warning.

#### AUTO settings

Without a target, these environment variables are passed to OpenVINO's AUTO device when the model is compiled. Each sets one OpenVINO property, and unset variables keep OpenVINO's defaults. `OPENCV_DNN_OPENVINO_CACHE_DIR` applies to every device.

| Variable | Example | OpenVINO property |
|---|---|---|
| `OPENCV_DNN_OPENVINO_AUTO_DEVICES` | `GPU,NPU,CPU` | `ov::device::priorities` |
| `OPENCV_DNN_OPENVINO_AUTO_UTILIZATION_THRESHOLD` | `GPU:80,CPU:90` | `ov::intel_auto::devices_utilization_threshold` |
| `OPENCV_DNN_OPENVINO_AUTO_PERF_CURVES` | `curves.json` | `ov::intel_auto::perf_curve_table` |
| `OPENCV_DNN_OPENVINO_AUTO_LOW_POWER_DEVICE` | `NPU` | `ov::intel_auto::low_power_device` |
| `OPENCV_DNN_OPENVINO_AUTO_COMPILE_FOR_ALL` | `1` | `ov::intel_auto::compile_for_all` |
| `OPENCV_DNN_OPENVINO_CACHE_DIR` | `/tmp/ov_cache` | `ov::cache_dir` |

- The device list gives the candidates in order of priority. Naming devices turns off AUTO's default exclusions, so the NPU can be chosen. On a machine that also has a non-Intel GPU, name the Intel GPU by its index, for example `GPU.0`.
- A device is left out while its utilization is above its threshold.
- The perf curve file gives, for each device key (`CPU`, `iGPU`, `dGPU`, `NPU`), a score at each utilization percent, and AUTO picks the device with the lowest score at its current utilization. The curves should cover 0 to 100 percent:
  ```json
  { "CPU": [[0, 9.4], [50, 15.0], [100, 30.0]], "iGPU": [[0, 13.9], [50, 20.0], [100, 60.0]] }
  ```
- The low power device is used while the platform runs in a low power mode.
- Compile for all compiles the model for the other candidates in the background and keeps them in the cache directory, which it needs, so that switching device does not wait for a compile.

The utilization thresholds, perf curves and low power device need the Intel Innovation Platform Framework (IPF) 2.5 or newer on the machine, which reports the utilization and power state; the low power mode also needs Intel Dynamic Tuning Technology. Without IPF, OpenVINO ignores these three settings and AUTO behaves as with its defaults. AUTO in OpenVINO 2026.4.1 uses them once, to choose the device when the model is compiled. OpenVINO versions with per-inference device selection choose again for every inference; inferences then run one at a time, without the first inferences on the CPU. Settings that the OpenVINO in use does not support are skipped with a warning. With `OPENCV_LOG_LEVEL=DEBUG`, AUTO also prints its own log, which shows every device switch.

The curves belong to one model on one platform and are made ahead of time, for example with Intel's xPU Stress Tool (xST).

### Selecting an Engine

The engine parameter is accepted by the `readNet*()` family of functions.

@add_toggle_cpp
@code{.cpp}
// ENGINE_AUTO is the default (resolves to ENGINE_OPENCV)
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");

// Explicitly select OpenCV's built-in engine
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_OPENCV);

// Use ONNX Runtime (ONNX models only, requires WITH_ONNXRUNTIME=ON)
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_ORT);

// Use the OpenVINO runtime (ONNX models only, requires WITH_OPENVINO=ON)
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_OPENVINO);
@endcode
@end_toggle

@add_toggle_python
@code{.py}
import cv2

# ENGINE_AUTO is the default (resolves to ENGINE_OPENCV)
net = cv2.dnn.readNetFromONNX("model.onnx")

# Explicitly select OpenCV's built-in engine
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_OPENCV)

# Use ONNX Runtime (ONNX models only, requires WITH_ONNXRUNTIME=ON)
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_ORT)

# Use the OpenVINO runtime (ONNX models only, requires WITH_OPENVINO=ON)
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_OPENVINO)
@endcode
@end_toggle

The engine cannot be changed after a model has been loaded. To use a different engine, the model must be reloaded with the new engine argument.

### Engine Selection via Environment Variable

The engine can be overridden at the process level using the `OPENCV_FORCE_DNN_ENGINE` environment variable. The integer values correspond directly to the `EngineType` enum: 0 for `ENGINE_AUTO`, 1 for `ENGINE_OPENCV`, 2 for `ENGINE_ORT`, and 3 for `ENGINE_OPENVINO`. Leaving the variable unset (or set to `ENGINE_AUTO`) does not force any engine, so the value passed to `readNet*()` is honored.

```bash
# Linux / macOS
OPENCV_FORCE_DNN_ENGINE=1 python3 inference.py

# Windows
set OPENCV_FORCE_DNN_ENGINE=1
python inference.py
```
