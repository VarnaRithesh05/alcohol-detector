@echo off
echo Starting Alcohol Detector Backend...
cd /d "%~dp0backend"
call npm install
node server.js
pause
