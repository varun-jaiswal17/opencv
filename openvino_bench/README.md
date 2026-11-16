# ENGINE_OPENVINO benchmark

This branch builds OpenCV against the **official OpenVINO 2026.4.1 release** and benchmarks
`ENGINE_OPENVINO`. A second branch, `engine-openvino-bench-ov-nightly`, has the same OpenCV code
and the same benchmark, built against an OpenVINO nightly package. Only `config.bat` differs
between the two branches.

## Steps

Run these from the repository root in a `cmd` window.

1. Download the OpenVINO package named in `config.bat` (`OV_URL`) and extract it into `OV_HOME`.
2. Download the benchmark models into opencv_extra (`xfeat.onnx` is already there):

   ```
   cd /d D:\Varun\opencv_extra\testdata\dnn
   python download_models.py --download_large_model yolo26n yolov10s
   ```

3. Build `opencv_perf_dnn` and `opencv_test_dnn` in a build folder of their own:

   ```
   openvino_bench\build.bat
   ```

4. Run the benchmark. The results go to `BENCH_RESULTS`, named after the OpenVINO version:

   ```
   openvino_bench\run.bat
   ```

5. Once both branches have been run, compare the results:

   ```
   openvino_bench\compare.bat
   ```

The paths in `config.bat` can be overridden by setting the same variables before running the
scripts.

## What is measured

| Test | What one sample covers |
|---|---|
| `DNN_OpenVINO_Engine.device_discovery` | a new `ov::Core` finds, loads and queries its plugins (`get_available_devices()`); it runs in a process of its own |
| `DNN_OpenVINO_Target.first_forward` | `readNetFromONNX()`, `setPreferableTarget()`, `setInput()` and the first `forward()`, which compiles the model for the device |
| `DNN_OpenVINO_Target.forward` | `setInput()` and `forward()` on a model that is already compiled |

Targets: `CPU`, `GPU_FP32` (`DNN_TARGET_OPENCL`), `GPU_FP16` (`DNN_TARGET_OPENCL_FP16`) and
`NPU`. A target whose device OpenVINO does not find is skipped, so it is never measured as the CPU
fallback. "GPU" is OpenVINO's first GPU, the Intel integrated GPU on a machine that has one.

Models (input shape checked with OpenVINO 2026.4.1):

| Model | Input | Released | Licence |
|---|---|---|---|
| `yolo26n.onnx` (Ultralytics YOLO26) | 1x3x640x640 | 2026 | AGPL-3.0 |
| `yolov10s.onnx` (YOLOv10, Tsinghua University) | 1x3x480x640 | 2024 | AGPL-3.0 |
| `xfeat.onnx` (XFeat) | 1x1x640x640 | 2024 | Apache-2.0 |

Each test records the OpenVINO version it was built with and the version of the `openvino.dll`
it loaded. The test fails when they differ, which happens when another OpenVINO comes first on
`PATH`.
