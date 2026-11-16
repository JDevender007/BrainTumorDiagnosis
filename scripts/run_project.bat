@echo off
title Brain Tumor Diagnosis - Full Pipeline
echo ============================================================
echo   BRAIN TUMOR DIAGNOSIS - FULL PROJECT RUNNER
echo   Tumor-Aware Self-Supervised Multi-Task Architecture
echo ============================================================
echo.

cd /d "%~dp0.."

echo [1/10] Checking environment...
python -c "import torch, torchvision, numpy, sklearn, matplotlib; print('  All dependencies OK')" || (
    echo ERROR: Missing dependencies. Run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [2/10] Checking datasets...
python -c "from pathlib import Path; p=Path('data/raw/classification'); print(f'  Internal: {sum(1 for _ in p.rglob(\"*.jpg\"))} images' if p.exists() else '  Internal: NOT FOUND')"

echo.
echo [3/10] Running full experiment pipeline (7 phases)...
echo   This trains models and evaluates them. May take 2-6 hours.
python scripts/run_research_experiments.py
if %errorlevel% neq 0 (
    echo   ERROR: Experiment pipeline failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [4/10] Running multi-seed validation (3 seeds)...
echo   Training proposed model 3 times for statistical validation.
python scripts/run_multi_seed.py
if %errorlevel% neq 0 (
    echo   WARNING: Multi-seed validation failed. Continuing...
)

echo.
echo [5/10] Generating 12 tables...
python scripts/generate_tables.py
if %errorlevel% neq 0 (
    echo   WARNING: Table generation failed. Continuing...
)

echo.
echo [6/10] Generating 16 figures...
python scripts/generate_figures.py
if %errorlevel% neq 0 (
    echo   WARNING: Figure generation failed. Continuing...
)

echo.
echo [7/10] Generating IEEE paper...
python scripts/generate_paper.py
if %errorlevel% neq 0 (
    echo   WARNING: Paper generation failed. Continuing...
)

echo.
echo [8/10] Exporting all outputs to one folder...
python scripts/export_all_outputs.py
if %errorlevel% neq 0 (
    echo   WARNING: Export failed. Continuing...
)

echo.
echo [9/10] Testing on Kaggle dataset...
python scripts/test_quick.py
if %errorlevel% neq 0 (
    echo   WARNING: Kaggle test failed. Continuing...
)

echo.
echo [10/10] Generating evaluation PNGs for Kaggle dataset...
python scripts/evaluate_kaggle_full.py
if %errorlevel% neq 0 (
    echo   WARNING: Kaggle evaluation failed. Continuing...
)

echo.
echo ============================================================
echo   ALL DONE!
echo ============================================================
echo.
echo   Output folders:
echo     outputs/research_experiments/all_outputs/  (42 files)
echo     outputs/kaggle_evaluation/                 (10 files)
echo.
echo   Run 'python scripts\run_everything.py' for interactive menu.
echo.
pause
