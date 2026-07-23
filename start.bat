@echo off
REM ====================================================================
REM  Mind-Control Console launcher
REM  Starts the mapper/console server for Qwen3.5-0.8B.
REM  Ctrl+C (or closing this window) shuts the server down gracefully
REM  and releases the GPU -- the Python process runs IN this console,
REM  so the interrupt reaches it directly.
REM ====================================================================

setlocal
cd /d "%~dp0"

set MAP=interpretability_lab\mapper\maps\qwen3_5_0_8b.json
set PORT=8000

if not exist "%MAP%" (
    echo [start] map not found: %MAP%
    echo [start] build it first ^(see README: mapper^), then re-run.
    pause
    exit /b 1
)

REM Pick a Python: prefer "py" launcher, fall back to "python".
where py >nul 2>nul && (set PY=py) || (set PY=python)

title Mind-Control Console  (Ctrl+C to stop)
echo [start] launching console on http://127.0.0.1:%PORT%
echo [start] the model takes ~30s to load; the browser will open when ready.

REM Open the browser after a short delay, in the background, without
REM blocking or spawning a process that outlives the server.
start "" /b cmd /c "timeout /t 35 /nobreak >nul & start http://127.0.0.1:%PORT%"

REM Run the server in THIS console window so Ctrl+C is delivered straight
REM to Python (uvicorn -> app "shutdown" event -> GPU released).
%PY% -m interpretability_lab.mapper.server --map "%MAP%" --port %PORT%

echo.
echo [start] console stopped.
endlocal
