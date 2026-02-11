@echo off
setlocal enabledelayedexpansion

REM -------- Activate conda env (adjust if your Anaconda path differs) --------
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
call conda activate torch_env

SET ANNOTATIONS=.\failure_analysis\analysis_all\annotations_clean.csv

echo ==========================================================
echo [START] Improvement Experiments (Prompt Ensembling + Pooling)
echo ==========================================================

REM ----------------------------------------------------------
REM 6.1 Object+Attribute (n~87)
REM ----------------------------------------------------------
echo [6.1] Object+Attribute...
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object,Attribute --max_subset 200 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object,Attribute --max_subset 200 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object,Attribute --max_subset 200 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM ----------------------------------------------------------
REM 6.2 Object (n~62)
REM ----------------------------------------------------------
echo [6.2] Object...
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object --max_subset 200 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object --max_subset 200 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Object --max_subset 200 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM ----------------------------------------------------------
REM 6.3 Attribute (n~25)
REM ----------------------------------------------------------
echo [6.3] Attribute...
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Attribute --max_subset 200 --do_prompt_ensemble --pooling max
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Attribute --max_subset 200 --do_prompt_ensemble --pooling mean
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Attribute --max_subset 200 --do_prompt_ensemble --pooling logsumexp --tau 1.0

REM ----------------------------------------------------------
REM 6.4 Action (n~24) + export hits for bootstrapping
REM ----------------------------------------------------------
echo [6.4] Action (export hits)...
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Action --max_subset 200 --do_prompt_ensemble --pooling max --save_hits_csv
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Action --max_subset 200 --do_prompt_ensemble --pooling mean --save_hits_csv
python improve_subset.py --annotations_csv %ANNOTATIONS% --include_categories Action --max_subset 200 --do_prompt_ensemble --pooling logsumexp --tau 1.0 --save_hits_csv

echo ----------------------------------------------------------
echo [7] Summarizing results JSON -> summary_subset_results.csv
echo ----------------------------------------------------------
python summarize_result.py

echo ----------------------------------------------------------
echo [8] Bootstrapping CI on Action (max / mean / logsumexp)
echo ----------------------------------------------------------
python bootstrap_ci.py --hits_csv subset_hits_Action_n24_max_seed42.csv --B 2000
python bootstrap_ci.py --hits_csv subset_hits_Action_n24_mean_seed42.csv --B 2000
python bootstrap_ci.py --hits_csv subset_hits_Action_n24_logsumexp_seed42.csv --B 2000

echo ==========================================================
echo [FINISH] Done. See summary_subset_results.csv and CI outputs above.
echo ==========================================================
pause
