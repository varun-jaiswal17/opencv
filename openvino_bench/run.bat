@echo off
rem Runs the ENGINE_OPENVINO benchmark of this branch's build and writes the results to
rem %BENCH_RESULTS%\discovery_<version>.xml and %BENCH_RESULTS%\engine_<version>.xml.
rem Extra arguments go to opencv_perf_dnn, for example --perf_min_samples=20.
setlocal
call "%~dp0config.bat" || exit /b 1

set "PERF=%BENCH_BUILD%\bin\opencv_perf_dnn.exe"
if not exist "%PERF%" (
    echo %PERF% not found, run build.bat first
    exit /b 1
)

rem puts this package's openvino.dll, plugins and tbb first on PATH
call "%OV_ROOT%\setupvars.bat" >nul || exit /b 1
set "OPENCV_TEST_DATA_PATH=%OPENCV_EXTRA%\testdata"
if not exist "%BENCH_RESULTS%" mkdir "%BENCH_RESULTS%"

rem device discovery runs in a process of its own, so no earlier test has loaded the plugins
"%PERF%" --gtest_filter=DNN_OpenVINO_Engine.device_discovery "--gtest_output=xml:%BENCH_RESULTS%\discovery_%OV_NAME%.xml" %* || exit /b 1
"%PERF%" --gtest_filter=DNN_OpenVINO_Target.* "--gtest_output=xml:%BENCH_RESULTS%\engine_%OV_NAME%.xml" %* || exit /b 1

echo Results of OpenVINO %OV_NAME% are in %BENCH_RESULTS%
