@echo off
rem OpenVINO package this branch builds and benchmarks against: the official 2026.4.1 release.
rem Any of these can be set in the environment beforehand to point somewhere else.

if not defined OV_NAME set "OV_NAME=2026.4.1"
if not defined OV_PACKAGE set "OV_PACKAGE=openvino_toolkit_windows_2026.4.1.22982.07f9c262b05_x86_64"
if not defined OV_URL set "OV_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4.1/windows/%OV_PACKAGE%.zip"

if not defined OV_HOME set "OV_HOME=D:\Varun\openvino"
if not defined OPENCV_EXTRA set "OPENCV_EXTRA=D:\Varun\opencv_extra"
if not defined BENCH_BUILD set "BENCH_BUILD=D:\Varun\build_ov_%OV_NAME%"
if not defined BENCH_RESULTS set "BENCH_RESULTS=D:\Varun\openvino_bench_results"
if not defined VSVARS set "VSVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "OV_ROOT=%OV_HOME%\%OV_PACKAGE%"
exit /b 0
