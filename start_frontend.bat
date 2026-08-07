@echo off
echo Starting React Dashboard...
cd /d "%~dp0frontend"
call npm install
npm start
pause
