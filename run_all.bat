@echo off
call conda activate torch_env

REM 1) Baseline / cache
python main.py

REM 2) failure assignment (overlap + A/B/C)
python failure.py

REM 3) analyze overlap annotations
cd failure_analysis
python analyze_annotations.py --csv assign_overlap.csv --outdir analysis_out
cd ..

REM 4) improve on overlap subset
python improve_subset.py --failure_samples_csv .\failure_analysis\assign_overlap.csv --include_categories Attribute,Object --max_subset 30 --do_prompt_ensemble

pause
