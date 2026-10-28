@echo off
setlocal

REM Double-click entry point for a new Windows machine.  The PowerShell helper
REM validates dependencies, requests consent before installing anything, and
REM launches ESPI only after a later clean validation run.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_espi_live.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Setup did not launch ESPI this time. Review the messages above, then run this file again.
    pause
)
endlocal & exit /b %EXIT_CODE%
