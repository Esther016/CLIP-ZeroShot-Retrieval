@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Post-annotation pipeline:
REM   4) Merge assignments (root/merge_assignments.py)
REM   5) Analyze -> failure_analysis/analysis_all/annotations_clean.csv
REM   6) Improve (prompt ensembling + pooling ablation)
REM   7) Summarize json -> summary_subset_results.csv
REM   8) Bootstrap CI (optional, requires bootstrap_ci.py)
REM ==========================================================

REM -------- Activate conda env (adjust if your Anaconda path differs) --------
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
call conda activate torch_env

REM -------- Paths --------
set ROOT=%CD%
set FAILDIR=%ROOT%\failure_analysis
set OUTDIR=%FAILDIR%\analysis_all
set MERGE_SCRIPT=%ROOT%\merge_assignments.py
set ANALYZE_SCRIPT=%FAILDIR%\analyze_annotations.py
set ANNOTATIONS=%OUTDIR%\annotations_clean.csv

echo ==========================================================
echo [START] Post-annotation pipeline
echo ==========================================================
echo [INFO] ROOT      = %ROOT%
echo [INFO] FAILDIR   = %FAILDIR%
echo [INFO] OUTDIR    = %OUTDIR%

REM -------- Sanity check: assignment files exist --------
echo ----------------------------------------------------------
echo [CHECK] Required assignment files
echo ----------------------------------------------------------
if not exist "%FAILDIR%\assign_A.csv" (echo [WARN] Missing assign_A.csv)
if not exist "%FAILDIR%\assign_B.csv" (echo [WARN] Missing assign_B.csv)
if not exist "%FAILDIR%\assign_C.csv" (echo [WARN] Missing assign_C.csv)
if not exist "%FAILDIR%\assign_overlap.csv" (echo [WARN] Missing assign_overlap.csv)

REM -------- 4) Merge assignments --------
echo ----------------------------------------------------------
echo [4] Merge assignments -> failure_analysis\assign_all*.csv
echo ----------------------------------------------------------
if not exist "%MERGE_SCRIPT%" (
  echo [ERROR] merge_assignments.py not found at: %MERGE_SCRIPT%
  echo         You said merge script is in repo root. Please place it there.
  exit /b 1
)
python "%MERGE_SCRIPT%"
if errorlevel 1 (
  echo [ERROR] merge_assignments failed.
  exit /b 1
)

REM -------- 5) Analyze annotations --------
echo ----------------------------------------------------------
echo [5] Analyze merged annotations -> failure_analysis\analysis_all\
echo ----------------------------------------------------------
if not exist "%ANALYZE_SCRIPT%" (
  echo [ERROR] analyze_annotations.py not found at: %ANALYZE_SCRIPT%
  exit /b 1
)

if not exist "%OUTDIR%" (
  mkdir "%OUTDIR%"
)

REM Use merged file produced by merge_assignments.py:
if not exist "%FAILDIR%\assign_all.csv" (
  echo [ERROR] Missing merged file: %FAILDIR%\assign_all.csv
  echo         Check merge_assignments.py outputs.
  exit /b 1
)

python "%ANALYZE_SCRIPT%" --csv "%FAILDIR%\assign_all.csv" --outdir "%OUTDIR%"
if errorlevel 1 (
  echo [ERROR] analyze_annotations failed.
  exit /b 1
)

REM Verify cleaned output exists
if not exist "%ANNOTATIONS%" (
  echo [ERROR] annotations_clean.csv not found at: %ANNOTATIONS%
  echo         analyze_annotations did not generate expected output.
  exit /b 1
)

echo [OK] Found cleaned annotations: %ANNOTATIONS%

REM -------- 6) Improvement experiments --------
echo ==========================================================
echo [6] Improvement Experiments (Prompt Ensembling + Pooling)
echo ==========================================================

REM 6.1 Object+Attribute
echo [6.1] Object+Attribute...
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object,Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object,Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object,Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM 6.2 Object
echo [6.2] Object...
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object --max_subset 200 --seed 42 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object --max_subset 200 --seed 42 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Object --max_subset 200 --seed 42 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM 6.3 Attribute
echo [6.3] Attribute...
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Attribute --max_subset 200 --seed 42 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM 6.4 Action (save hits for bootstrap if supported)
echo [6.4] Action (save hits if supported)...
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Action --max_subset 200 --seed 42 --do_prompt_ensemble --pooling max --save_hits_csv
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Action --max_subset 200 --seed 42 --do_prompt_ensemble --pooling mean --save_hits_csv
python improve_subset.py --annotations_csv "%ANNOTATIONS%" --include_categories Action --max_subset 200 --seed 42 --do_prompt_ensemble --pooling logsumexp --tau 1.0 --save_hits_csv

REM -------- 7) Summarize --------
echo ----------------------------------------------------------
echo [7] Summarize results JSON -> summary_subset_results.csv
echo ----------------------------------------------------------
python summarize_result.py
if errorlevel 1 (
  echo [ERROR] summarize_result.py failed.
  exit /b 1
)

REM -------- 8) Bootstrap CI (optional) --------
echo ----------------------------------------------------------
echo [8] Bootstrapping CI (optional)
echo ----------------------------------------------------------
if exist "%ROOT%\bootstrap_ci.py" (
  if exist "%ROOT%\subset_hits_Action_n24_max_seed42.csv" (
    python bootstrap_ci.py --hits_csv subset_hits_Action_n24_max_seed42.csv --B 2000
  ) else (
    echo [SKIP] Missing subset_hits_Action_n24_max_seed42.csv
  )

  if exist "%ROOT%\subset_hits_Action_n24_mean_seed42.csv" (
    python bootstrap_ci.py --hits_csv subset_hits_Action_n24_mean_seed42.csv --B 2000
  ) else (
    echo [SKIP] Missing subset_hits_Action_n24_mean_seed42.csv
  )

  if exist "%ROOT%\subset_hits_Action_n24_logsumexp_seed42.csv" (
    python bootstrap_ci.py --hits_csv subset_hits_Action_n24_logsumexp_seed42.csv --B 2000
  ) else (
    echo [SKIP] Missing subset_hits_Action_n24_logsumexp_seed42.csv
  )
) else (
  echo [SKIP] bootstrap_ci.py not found. Skipping CI.
)

echo ==========================================================
echo [FINISH] Done.
echo   - %OUTDIR%\annotations_clean.csv
echo   - summary_subset_results.csv
echo ==========================================================
pause
