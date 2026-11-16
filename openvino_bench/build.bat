@echo off
rem Configures and builds opencv_perf_dnn and opencv_test_dnn against the OpenVINO package
rem named in config.bat, in a build folder used only for that package.
setlocal
call "%~dp0config.bat" || exit /b 1
set "SRC=%~dp0.."

if not exist "%OV_ROOT%\setupvars.bat" (
    echo OpenVINO package not found: %OV_ROOT%
    echo Download %OV_URL%
    echo and extract it into %OV_HOME%
    exit /b 1
)

where cl >nul 2>nul || call "%VSVARS%" || exit /b 1
call "%OV_ROOT%\setupvars.bat" || exit /b 1

cmake -S "%SRC%" -B "%BENCH_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_LIST=dnn,ts -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON ^
    -DWITH_OPENVINO=ON -DOpenVINO_DIR="%OV_ROOT%\runtime\cmake" ^
    -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DBUILD_EXAMPLES=OFF || exit /b 1

rem the build folder must use this branch's package, not one cached from an earlier configure
findstr /b /c:"OpenVINO_DIR:" "%BENCH_BUILD%\CMakeCache.txt" | findstr /c:"%OV_PACKAGE%" >nul || (
    echo %BENCH_BUILD% is not configured with %OV_PACKAGE%, use an empty BENCH_BUILD folder
    exit /b 1
)

cmake --build "%BENCH_BUILD%" --target opencv_perf_dnn opencv_test_dnn || exit /b 1
echo Built against OpenVINO %OV_NAME% in %BENCH_BUILD%
