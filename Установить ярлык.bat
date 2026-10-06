@echo off
rem Re-creates the "Transcriber" desktop shortcut (start.bat already does it after the installation).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"
if errorlevel 1 (
    echo Could not create the shortcut. Run start.bat directly.
) else (
    echo The "Transcriber" shortcut is on your desktop.
)
pause
