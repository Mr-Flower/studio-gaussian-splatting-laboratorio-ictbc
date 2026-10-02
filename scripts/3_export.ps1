# Passo 3 - Esporta l'ultimo training della scena in exports\<Scene>\splat.ply,
# da aprire in SuperSplat (https://superspl.at/editor) per pulizia e ritaglio.
# Uso:  .\scripts\3_export.ps1 -Scene arco
param([string]$Scene = 'arco')
. "$PSScriptRoot\..\activate.ps1"
$config = Get-ChildItem "$PSScriptRoot\..\outputs\$Scene" -Recurse -Filter config.yml |
    Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $config) { throw "Nessun training trovato in outputs\$Scene" }
"Config: $($config.FullName)"
ns-export gaussian-splat --load-config $config.FullName --output-dir "$PSScriptRoot\..\exports\$Scene"
if ($LASTEXITCODE -ne 0) { throw "ns-export fallito (exit $LASTEXITCODE)" }
