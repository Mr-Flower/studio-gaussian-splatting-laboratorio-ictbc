@echo off
rem Avvia l'interfaccia grafica (senza finestra di console).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Il programma non e' ancora installato: eseguire prima Installa.bat
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m app
