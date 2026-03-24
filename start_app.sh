#!/bin/bash
python3 -m venv venv
source venv/bin/activate
git pull
pip install -r requirements.txt
CORKYSOFT_ENV=development CORKYSOFT_ALLOW_ANONYMOUS_UI=1 streamlit run dashboard/app.py
