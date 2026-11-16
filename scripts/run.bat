@echo off
title Quick Run
cd /d "%~dp0.."
echo Running full pipeline...
python scripts\run_everything.py
pause
