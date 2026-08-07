@echo off
echo Starting Python Detection Engine...
cd /d "%~dp0python-engine"
pip install -r requirements.txt
python detector.py
pause
