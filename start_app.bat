@echo off
REM Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Pull latest code
git pull

REM Install/update dependencies
pip install -r requirements.txt

REM Run Streamlit app
set CORKYSOFT_ENV=development
set CORKYSOFT_ALLOW_ANONYMOUS_UI=1
streamlit run dashboard/app.py
