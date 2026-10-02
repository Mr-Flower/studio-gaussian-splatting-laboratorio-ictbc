@echo off
rem Installa tutto il necessario nella cartella del programma (non servono diritti di amministratore).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installa.ps1"
echo.
pause
