@echo off

:: 160x120 resolution, 10 samples per pixel, 2 threads
set RUN_CMD=win32-raytracer.exe 160 120 10 2 perfTest
set BUILD_CMD=msbuild win32-raytracer.sln -maxcpucount /p:Configuration=Release /p:Platform=x64
set PERF_FILE=perf.txt
set EXEC_PATH=x64\Release

:: Test the previous build
git stash
%BUILD_CMD%
pushd %EXEC_PATH%
%RUN_CMD%
popd
COPY %EXEC_PATH%\%PERF_FILE% prevPerf.txt

:: Test the current build
git stash pop
%BUILD_CMD%
pushd %EXEC_PATH%
%RUN_CMD%
popd
COPY %EXEC_PATH%\%PERF_FILE% currPerf.txt
