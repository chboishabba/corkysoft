#!/bin/bash
python3 -m venv venv
source venv/bin/activate
git pull
pip install -r requirements.txt
CORKYSOFT_ENV=development \
CORKYSOFT_REQUIRE_UI_AUTH=1 \
CORKYSOFT_AUTO_PROVISION_GOOGLE_ADMIN=1 \
streamlit run dashboard/app.py
