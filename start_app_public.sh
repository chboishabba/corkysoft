#!/bin/bash
python3 -m venv venv
source venv/bin/activate
git pull
pip install -r requirements.txt
# Run Streamlit in the background and tunnel the default port (8501)
CORKYSOFT_ENV=production streamlit run dashboard/app.py & ssh -R 80:localhost:8501 localhost.run
