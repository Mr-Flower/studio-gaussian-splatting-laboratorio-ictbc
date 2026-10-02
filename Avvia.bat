@echo off
rem Avvia l'interfaccia grafica (senza finestra di console).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m app
