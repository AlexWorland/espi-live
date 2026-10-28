@echo off
setlocal enabledelayedexpansion

set "SCRIPT_NAME=%~nx0"
set "EXIT_CODE=0"
set "DEFAULT_PY_DIR=C:\\Python310\\PySpin"
set "PYTHON_CMD=py -3.10"
set "WHEEL_FILE=spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl"
set "PAUSE_AT_END=1"
set "TARGET_PY_DIR=%DEFAULT_PY_DIR%"

if "%~1"=="" goto run

:parse_args
if "%~1"=="" goto run
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
if /I "%~1"=="/?" goto show_help
if /I "%~1"=="--dir" (
    shift
    if "%~1"=="" goto bad_args
    set "TARGET_PY_DIR=%~1"
    shift
    goto parse_args
)
if /I "%~1"=="--python" (
    shift
    if "%~1"=="" goto bad_args
    set "PYTHON_CMD=%~1"
    shift
    goto parse_args
)
if /I "%~1"=="--wheel" (
    shift
    if "%~1"=="" goto bad_args
    set "WHEEL_FILE=%~1"
    shift
    goto parse_args
)
if /I "%~1"=="--no-pause" (
    set "PAUSE_AT_END="
    shift
    goto parse_args
)
echo.
echo [ERROR] Unknown option: %~1
goto bad_args

:show_help
echo Ensures the Spinnaker/PySpin environment is ready.
echo.
echo Usage: %SCRIPT_NAME% [--dir PATH] [--python EXE] [--wheel FILE] [--no-pause]
echo.
echo   --dir PATH      Directory that contains python.exe and the wheel (default: %DEFAULT_PY_DIR%)
echo   --python EXE    Python executable to run the pip commands (default: %PYTHON_CMD%)
echo   --wheel FILE    Wheel filename relative to the directory (default: %WHEEL_FILE%)
echo   --no-pause      Do not pause when the script finishes.
echo   -h, --help      Show this help message and exit.
endlocal
exit /b 0

:bad_args
echo Run "%SCRIPT_NAME% --help" for usage information.
endlocal
exit /b 64

:run
if not exist "%TARGET_PY_DIR%" (
    echo.
    echo [ERROR] Could not find directory: %TARGET_PY_DIR%
    goto fail
)
pushd "%TARGET_PY_DIR%" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Unable to enter directory: %TARGET_PY_DIR%
    goto fail
)

echo.
echo [INFO] Using python command: %PYTHON_CMD%
call :run_cmd "%PYTHON_CMD% -m ensurepip" || goto fail
call :run_cmd "%PYTHON_CMD% -m pip install --upgrade pip numpy matplotlib" || goto fail
call :run_cmd "%PYTHON_CMD% -m pip install Pillow==9.2.0" || goto fail

if not exist "%WHEEL_FILE%" (
    echo.
    echo [ERROR] Wheel not found: %TARGET_PY_DIR%\%WHEEL_FILE%
    goto fail
)
call :run_cmd "%PYTHON_CMD% -m pip install %WHEEL_FILE%" || goto fail

echo.
echo [SUCCESS] PySpin installation commands completed.
popd >nul 2>&1
goto end

:run_cmd
echo.
echo ^> %~1
%~1
if errorlevel 1 (
    echo [ERROR] Command failed: %~1
    exit /b 1
)
exit /b 0

:fail
popd >nul 2>&1
echo.
echo [FAILURE] See errors above.
set "EXIT_CODE=1"
goto end

:end
if defined PAUSE_AT_END (
    echo.
    pause
)
endlocal & exit /b %EXIT_CODE%
