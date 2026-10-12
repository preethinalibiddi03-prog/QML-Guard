@echo off
title QML-Guard Launcher
cd /d "%~dp0"
echo ======================================================
echo Launching QML-Guard: Quantum Anomaly Lab...
echo ======================================================
python -m streamlit run app.py
pause
