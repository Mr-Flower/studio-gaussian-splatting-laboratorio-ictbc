# Attiva l'ambiente del progetto: venv Python + COLMAP, FFmpeg e Git portabili in tools\.
# Uso (da PowerShell, nella cartella del progetto):  . .\activate.ps1
$root = $PSScriptRoot
$env:PATH = @(
    "$root\tools\colmap-4.2.1\bin",
    "$root\tools\ffmpeg-9.0.2-essentials_build\bin",
    "$root\tools\git\cmd",
    $env:PATH
) -join ';'
# Evita problemi di encoding nell'output di nerfstudio (rich) su console Windows.
$env:PYTHONUTF8 = '1'
. "$root\.venv\Scripts\Activate.ps1"
