@echo off
setlocal
set "PY_DIR=C:\Python310\PySpin"
set "UTIL_DIR=%~dp0"
for %%I in ("%UTIL_DIR%..\..") do set "REPO_ROOT=%%~fI\"
set "SCRIPT=%REPO_ROOT%third_party\flir-spinnaker-4.2\Examples\Python3\AcquireAndDisplay.py"

if not exist "%PY_DIR%" (
    echo [ERROR] Directory not found: %PY_DIR%
    exit /b 1
)

pushd "%PY_DIR%" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Unable to enter directory: %PY_DIR%
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Example not found: %SCRIPT%
    exit /b 1
)

echo Running AcquireAndDisplay from %PY_DIR%
py -3.10 "%SCRIPT%" %*
set "EXIT_CODE=%errorlevel%"

popd >nul
exit /b %EXIT_CODE%
