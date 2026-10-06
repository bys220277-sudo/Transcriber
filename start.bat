@echo off
setlocal
title Transcriber - server (closes by itself with the app window)
cd /d "%~dp0"

rem Libraries live in a short path outside the program folder: deep unzip folders would
rem exceed the Windows 260-character path limit during installation.
if not defined TRANSCRIBER_VENV set "TRANSCRIBER_VENV=%LOCALAPPDATA%\Transcriber\venv"
set "VENV=%TRANSCRIBER_VENV%"

rem --- Find a system Python ---
set "SYSPY="
for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined SYSPY (
        "%%p" --version >nul 2>nul
        if not errorlevel 1 set "SYSPY=%%p"
    )
)

rem --- Installed copy ---
if exist "%VENV%\.ready" (
    set "PY=%VENV%\Scripts\python.exe"
    goto run
)

if not defined SYSPY (
    echo [Transcriber] Python not found.
    echo Install Python 3.10+ from https://www.python.org and check "Add Python to PATH",
    echo then run start.bat again.
    pause
    exit /b 1
)

rem --- First run: own venv + libraries (in a normal window, the shortcut starts this one minimized) ---
if not defined TRANSCRIBER_SETUP (
    set "TRANSCRIBER_SETUP=1"
    start "Transcriber setup" "%~f0"
    exit /b 0
)
echo [Transcriber] First run: installing, this takes a few minutes. Please wait...
echo.
if not exist "%VENV%\Scripts\python.exe" (
    "%SYSPY%" -m venv "%VENV%"
    if errorlevel 1 goto setup_failed
)
set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%PY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto setup_failed

rem The best model for this computer is downloaded now, so the program is ready right away:
rem large-v3 with an NVIDIA GPU, small (fast enough on a CPU) without one. Others download on first use.
set "MODEL=small"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    set "MODEL=large-v3"
    echo.
    echo [Transcriber] NVIDIA GPU found: installing CUDA libraries, about 600 MB...
    "%PY%" -m pip install -r requirements-gpu.txt --disable-pip-version-check
    if errorlevel 1 echo [Transcriber] CUDA libraries were not installed: the program will run on the CPU.
)

echo.
echo [Transcriber] Downloading the speech recognition model "%MODEL%"...
if "%MODEL%"=="large-v3" (echo It is about 3 GB, this is the longest step.) else (echo It is about 460 MB.)
"%PY%" app.py --download %MODEL%
if errorlevel 1 echo [Transcriber] The model was not downloaded now: it will download on the first transcription.

echo ok> "%VENV%\.ready"
if not defined TRANSCRIBER_NO_SHORTCUT powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1" >nul 2>nul
echo.
echo [Transcriber] Done. The "Transcriber" shortcut is on your desktop. Starting...

:run
"%PY%" app.py
if errorlevel 1 (
    echo.
    echo [Transcriber] Stopped with an error, see the messages above.
    pause
)
exit /b 0

:setup_failed
echo.
echo [Transcriber] Installation failed, see the messages above.
echo Check the internet connection and run start.bat again.
pause
exit /b 1
