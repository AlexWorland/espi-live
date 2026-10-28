@echo off
setlocal

set "ZIP_PATH=%~dp0..\..\vendor-flir-spinnaker\SpinnakerSDK_FULL_4.2.0.88_x64 (4).zip"
set "DEST_DIR=%TEMP%\SpinnakerSDKInstaller"

if not exist "%ZIP_PATH%" (
    echo [ERROR] Could not find %ZIP_PATH%
    goto :end
)

echo [INFO] Extracting "%ZIP_PATH%" to "%DEST_DIR%" and launching the installer...
powershell -NoProfile -ExecutionPolicy Bypass ^
    -Command ^
    "param([string]`$zip,[string]`$dest) ^
    if (Test-Path `$dest) { Remove-Item -Recurse -Force `$dest } ^
    New-Item -ItemType Directory -Path `$dest | Out-Null ^
    Expand-Archive -Path `$zip -DestinationPath `$dest -Force ^
    `$exe = Join-Path `$dest 'SpinnakerSDK_FULL_4.2.0.88_x64.exe' ^
    if (-not (Test-Path `$exe)) { Write-Error \"Installer not found: `$exe\"; exit 1 } ^
    Start-Process -FilePath `$exe -Verb RunAs -Wait" ^
    "%ZIP_PATH%" "%DEST_DIR%"

if errorlevel 1 (
    echo [ERROR] Installer failed or was cancelled.
) else (
    echo [SUCCESS] Spinnaker SDK installer finished.
)

:end
echo.
pause
endlocal
