@echo off
REM ==========================================
REM LLM Judge - Complete Workflow
REM From raw CSV to trained model
REM ==========================================

chcp 65001 >nul 2>&1

echo.
echo ==========================================
echo LLM Judge - Complete Workflow
echo ==========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: Virtual environment not found at venv\Scripts\
    echo Continuing with system Python...
)
echo.

REM Create directories
echo Creating directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist results mkdir results
if not exist logs mkdir logs
if not exist tools mkdir tools
echo Directories ready.
echo.

REM Check OpenAI API Key
echo Checking OpenAI API key...
if "%OPENAI_API_KEY%"=="" (
    echo.
    echo WARNING: OPENAI_API_KEY environment variable not set
    echo You need this for auto-labeling with GPT-4
    echo.
    echo To set it, run:
    echo   set OPENAI_API_KEY=sk-your-key-here
    echo.
    set /p continue="Continue anyway? (y/n): "
    if not "%continue%"=="y" (
        echo Exiting...
        pause
        exit /b 1
    )
) else (
    echo API key found.
)
echo.

REM Check required CSV files
echo Checking input CSV files...
if not exist data\answers_extraction.csv (
    echo ERROR: data\answers_extraction.csv not found
    echo Please place your CSV files in the data\ directory
    pause
    exit /b 1
)
if not exist data\answers_extraction_without_access.csv (
    echo ERROR: data\answers_extraction_without_access.csv not found
    echo Please place your CSV files in the data\ directory
    pause
    exit /b 1
)
echo Input files found.
echo.

REM ==========================================
REM STEP 1: Auto-labeling with GPT-4
REM ==========================================
echo ==========================================
echo STEP 1: Auto-Labeling with GPT-4
echo ==========================================
echo.
echo This will:
echo   - Use GPT-4 to judge attack success
echo   - Process ~3000 samples
echo   - Take 1-2 hours
echo   - Cost approximately $0.64 USD
echo.
set /p continue="Continue with auto-labeling? (y/n): "
if not "%continue%"=="y" (
    echo Skipping auto-labeling...
    goto :PREPROCESS
)

echo.
echo Running batch labeling...
python tools\auto_label.py --mode both --model gpt-4o-mini --batch-size 10 --delay 0.5
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Auto-labeling failed
    echo Check the error messages above
    pause
    exit /b 1
)
echo.
echo Auto-labeling completed successfully!
echo.

REM ==========================================
REM STEP 2: Data Preprocessing
REM ==========================================
:PREPROCESS
echo ==========================================
echo STEP 2: Data Preprocessing
echo ==========================================
echo.
echo Merging and splitting datasets...
python tools\preprocess_data.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Preprocessing failed
    pause
    exit /b 1
)
echo.
echo Preprocessing completed!
echo.

REM ==========================================
REM STEP 3: Train Model
REM ==========================================
echo ==========================================
echo STEP 3: Training Judge Model
echo ==========================================
echo.
echo Training with cross-validation...
python src\judge\train.py --data data\train.csv --out models\judge_clf.joblib --cross-validate --show-feature-importance --batch-size 32 > logs\training.log 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed
    echo Check logs\training.log for details
    pause
    exit /b 1
)
echo.
echo Training completed!
echo See logs\training.log for details
echo.

REM ==========================================
REM STEP 4: Evaluate Model
REM ==========================================
echo ==========================================
echo STEP 4: Evaluating Model
echo ==========================================
echo.
echo Running evaluation...
python src\judge\evaluate.py --data data\test.csv --model models\judge_clf.joblib --output-dir results --plot --error-analysis --save-predictions --batch-size 32 > logs\evaluation.log 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Evaluation failed
    echo Check logs\evaluation.log for details
    pause
    exit /b 1
)
echo.
echo Evaluation completed!
echo See logs\evaluation.log for details
echo.

REM ==========================================
REM STEP 5: Run Tests (Optional)
REM ==========================================
echo ==========================================
echo STEP 5: Running Tests (Optional)
echo ==========================================
echo.
set /p runtests="Run attack suite and performance tests? (y/n): "
if not "%runtests%"=="y" goto :SUMMARY

echo.
echo Running attack suite...
python tests\test_attack_suite.py --model models\judge_clf.joblib --risk-level medium > logs\attack_suite.log 2>&1

echo Running performance benchmark...
python tests\test_performance.py --model models\judge_clf.joblib --n-samples 500 > logs\performance.log 2>&1
echo Tests completed!
echo.

REM ==========================================
REM SUMMARY
REM ==========================================
:SUMMARY
echo.
echo ==========================================
echo WORKFLOW COMPLETED SUCCESSFULLY!
echo ==========================================
echo.
echo Generated Files:
echo ----------------
echo Data:
echo   - data\answers_extraction_labeled.csv
echo   - data\answers_extraction_without_access_labeled.csv
echo   - data\combined_labeled.csv
echo   - data\train.csv
echo   - data\test.csv
echo.
echo Model:
echo   - models\judge_clf.joblib
echo.
echo Results:
echo   - results\roc_curve.png
echo   - results\pr_curve.png
echo   - results\error_cases.csv
echo   - results\predictions.csv
echo.
echo Logs:
echo   - logs\training.log
echo   - logs\evaluation.log
echo   - logs\attack_suite.log (if run)
echo   - logs\performance.log (if run)
echo.
echo Next Steps:
echo -----------
echo 1. View evaluation metrics:
echo    type logs\evaluation.log | findstr "Accuracy F1"
echo.
echo 2. Open results folder:
echo    explorer results
echo.
echo 3. Check error cases:
echo    type results\error_cases.csv
echo.
echo 4. View training details:
echo    type logs\training.log
echo.
pause
