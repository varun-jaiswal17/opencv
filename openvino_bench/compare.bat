@echo off
rem Puts the results of every OpenVINO version in %BENCH_RESULTS% side by side, as median
rem times in milliseconds. The first file of each table is the reference for the ratio columns.
setlocal
call "%~dp0config.bat" || exit /b 1
set "SUMMARY=%~dp0..\modules\ts\misc\summary.py"

echo Device discovery
python "%SUMMARY%" -o txt -m median -u ms "%BENCH_RESULTS%\discovery_*.xml" || exit /b 1
echo.
echo First forward (read, compile, one inference) and forward
python "%SUMMARY%" -o txt -m median -u ms "%BENCH_RESULTS%\engine_*.xml" || exit /b 1
