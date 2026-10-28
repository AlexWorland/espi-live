@echo off
setlocal

REM Always open the project root (two levels above this script) in VS Code.
for %%I in ("%~dp0..\..") do set "targetDir=%%~fI"

set "codeExe=%LocalAppData%\Programs\Microsoft VS Code\Code.exe"
if exist "%ProgramFiles%\Microsoft VS Code\Code.exe" set "codeExe=%ProgramFiles%\Microsoft VS Code\Code.exe"
if exist "%ProgramFiles(x86)%\Microsoft VS Code\Code.exe" set "codeExe=%ProgramFiles(x86)%\Microsoft VS Code\Code.exe"

if not exist "%codeExe%" (
    set "codeExe=code"
)

REM Launch VS Code without keeping this console window open.
start "" "%codeExe%" "%targetDir%"

endlocal
exit /b 0
