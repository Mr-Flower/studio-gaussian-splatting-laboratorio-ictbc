# Run definitivi per l'articolo (fase 6 di PIANO_ARTICOLO.md), uno dopo l'altro.
# Uso, dalla cartella del programma:
#   powershell -ExecutionPolicy Bypass -File scripts\run_articolo.ps1 [-Progetto foto] [-Riusa]
# Con -Riusa non si rifanno i run di Inria e di splatfacto-big gia' presenti alla qualita' massima.
# Ogni comando continua anche se il precedente non riesce; l'esito di ciascuno e' nel registro
# data\<progetto>\run_articolo.log.
param(
    [string]$Progetto = 'foto',
    [switch]$Riusa
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$registro = Join-Path $root "data\$Progetto\run_articolo.log"
$env:PYTHONUTF8 = '1'

function Esegui([string]$titolo, [string[]]$argomenti) {
    $inizio = Get-Date
    Add-Content -Path $registro -Encoding utf8 -Value "`n### $($inizio.ToString('yyyy-MM-dd HH:mm:ss')) $titolo"
    # Il dettaglio di ogni passo e' gia' in data\<progetto>\pipeline.log: qui solo l'uscita breve.
    & $py @argomenti 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $registro -Append -Encoding utf8
    $esito = $LASTEXITCODE
    $minuti = [math]::Round(((Get-Date) - $inizio).TotalMinutes, 1)
    Add-Content -Path $registro -Encoding utf8 -Value "### fine: codice $esito, $minuti minuti"
}

$run = @('-m', 'app.cli', 'run', '--project', $Progetto, '--steps', 'train', 'eval', 'export')

# 1. Confronto principale: risoluzione piena, pose e aspetto spenti (default).
$principali = if ($Riusa) { @('splatfacto', 'nerfacto', 'nerfacto-big') } else { @('splatfacto', 'splatfacto-big', 'inria-3dgs', 'nerfacto', 'nerfacto-big') }
Esegui 'Confronto principale (qualita'' massima)' ($run + @('--quality', 'massima', '--methods') + $principali)

# 2. nerfacto con le opzioni originali di nerfstudio, per misurarne l'effetto.
Esegui 'nerfacto con pose e aspetto' ($run + @('--quality', 'massima', '--methods', 'nerfacto', '--pose-refinement', '--appearance-embedding'))

# 3. Sensibilita' alla risoluzione dei due 3DGS.
Esegui '3DGS a qualita'' alta' ($run + @('--quality', 'alta', '--methods', 'splatfacto', 'inria-3dgs'))
Esegui '3DGS a qualita'' media' ($run + @('--quality', 'media', '--methods', 'splatfacto', 'inria-3dgs'))

# 4. Variabilita' tra esecuzioni: due semi in piu' alla qualita' media.
foreach ($seme in 1, 2) {
    Esegui "3DGS a qualita' media, seme $seme" ($run + @('--quality', 'media', '--methods', 'splatfacto', 'inria-3dgs', '--seed', "$seme"))
}

# 5. Controllo delle immagini di riferimento, report e materiali dell'articolo.
Esegui 'Controllo delle immagini di riferimento' @('-m', 'app.check_gt', '--project', $Progetto)
Esegui 'Report' @('-m', 'app.report', '--project', $Progetto)
Esegui 'Materiali per l''articolo' @('-m', 'app.paper', '--project', $Progetto, '--out', "paper\$Progetto", '--scartate', 'foto\LEGGIMI.txt')
Add-Content -Path $registro -Encoding utf8 -Value "`n### tutto finito $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
