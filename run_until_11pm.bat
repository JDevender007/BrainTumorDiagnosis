@echo off
REM ===========================================================================
REM  run_until_11pm.bat - item-G retrain queue (intensity aug + per-image norm)
REM
REM  Runs the whole task list, then STOPS all compute at 23:00 local time:
REM    seeds 42..46: train 30 epochs (batch 8, 2 at a time) -> TEST eval once
REM    then:         multi-seed aggregate -> 27x3 corruption MRRS
REM
REM  Nothing is lost at the stop: training resumes from checkpoint_last.pt,
REM  finished evals/aggregates are skipped. Just run this file again later to
REM  continue exactly where it stopped.
REM
REM  Usage:
REM    run_until_11pm.bat              ^<- stops at 23:00, 2 trainings at once
REM    run_until_11pm.bat 22:30        ^<- custom stop time (HH:MM)
REM    run_until_11pm.bat 23:00 1      ^<- one training at a time
REM    run_until_11pm.bat 23:00 3      ^<- 3 at once (RAM-tight, repo max)
REM
REM  Logs: each stage prints to this console; the full transcript also goes to
REM  logs\run_until_11pm.log (gitignored).
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "DEADLINE=%~1"
if "%DEADLINE%"=="" set "DEADLINE=23:00"

set "PARALLEL=%~2"
if "%PARALLEL%"=="" set "PARALLEL=2"

if not exist "logs" mkdir "logs"

echo.
echo ============================================================
echo  TAMNet item-G retrain - starts %DATE% %TIME%, stops at %DEADLINE%
echo  Re-run this file to resume. Parallel trainings: %PARALLEL%
echo ============================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "python -X utf8 scripts\run_until_deadline.py --deadline '%DEADLINE%' --parallel %PARALLEL% 2>&1 | Tee-Object -FilePath 'logs\run_until_11pm.log' -Append; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"
echo.
echo ============================================================
echo  run_until_11pm.bat finished %DATE% %TIME% (exit %RC%)
if not "%RC%"=="0" echo  A stage failed - nothing was deleted; fix it and run this file again.
echo ============================================================
endlocal & exit /b %RC%
