# Run definitivi per l'articolo (fase 6 di PIANO_ARTICOLO.md), uno dopo l'altro.
# Uso, dalla cartella del programma:
#   powershell -ExecutionPolicy Bypass -File scripts\run_articolo.ps1 [-Progetto foto] [-Riusa]
# Con -Riusa non si rifanno i run di Inria e di splatfacto-big gia' presenti alla qualita' massima.
# Per riprendere dopo un'interruzione:
#   -SoloValutazione m1,m2  metodi gia' allenati di cui fare solo valutazione ed esportazione
#   -Principali m1,m2       metodi da allenare nel confronto principale (al posto di quelli predefiniti)
#   -Da N                   salta i passi prima del numero N (1 = confronto principale ... 9 = materiali)
# Ogni comando continua anche se il precedente non riesce; l'esito di ciascuno e' nel registro
# data\<progetto>\run_articolo.log.
param(
    [string]$Progetto = 'foto',
    [switch]$Riusa,
    [string[]]$SoloValutazione = @(),
    [string[]]$Principali = @(),
    [int]$Da = 1
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$registro = Join-Path $root "data\$Progetto\run_articolo.log"
$env:PYTHONUTF8 = '1'
$script:passo = 0

function Esegui([string]$titolo, [string[]]$argomenti) {
    $inizio = Get-Date
    Add-Content -Path $registro -Encoding utf8 -Value "`n### $($inizio.ToString('yyyy-MM-dd HH:mm:ss')) $titolo"
    # Il dettaglio di ogni passo e' gia' in data\<progetto>\pipeline.log: qui solo l'uscita breve.
    & $py @argomenti 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $registro -Append -Encoding utf8
    $esito = $LASTEXITCODE
    $minuti = [math]::Round(((Get-Date) - $inizio).TotalMinutes, 1)
    Add-Content -Path $registro -Encoding utf8 -Value "### fine: codice $esito, $minuti minuti"
}

# Vero se il passo numerato va eseguito (cioe' non precede quello indicato con -Da).
function Passo {
    $script:passo++
    return $script:passo -ge $Da
}

$run = @('-m', 'app.cli', 'run', '--project', $Progetto, '--steps', 'train', 'eval', 'export')

if ($SoloValutazione.Count -gt 0) {
    Esegui 'Ripresa: valutazione ed esportazione' (@('-m', 'app.cli', 'run', '--project', $Progetto, '--steps', 'eval', 'export', '--quality', 'massima', '--methods') + $SoloValutazione)
}

# 1. Confronto principale: risoluzione piena, pose e aspetto spenti (default).
$tutti = @('splatfacto', 'splatfacto-big', 'inria-3dgs', 'nerfacto', 'nerfacto-big')
$principali = if ($Principali.Count -gt 0) { $Principali } elseif ($Riusa) { @('splatfacto', 'nerfacto', 'nerfacto-big') } else { $tutti }
if (Passo) { Esegui 'Confronto principale (qualita'' massima)' ($run + @('--quality', 'massima', '--methods') + $principali) }

# 2. nerfacto con le opzioni originali di nerfstudio, per misurarne l'effetto.
if (Passo) { Esegui 'nerfacto con pose e aspetto' ($run + @('--quality', 'massima', '--methods', 'nerfacto', '--pose-refinement', '--appearance-embedding')) }

# 3-4. Sensibilita' alla risoluzione dei due 3DGS.
if (Passo) { Esegui '3DGS a qualita'' alta' ($run + @('--quality', 'alta', '--methods', 'splatfacto', 'inria-3dgs')) }
if (Passo) { Esegui '3DGS a qualita'' media' ($run + @('--quality', 'media', '--methods', 'splatfacto', 'inria-3dgs')) }

# 5-6. Variabilita' tra esecuzioni: due semi in piu' alla qualita' media.
foreach ($seme in 1, 2) {
    if (Passo) { Esegui "3DGS a qualita' media, seme $seme" ($run + @('--quality', 'media', '--methods', 'splatfacto', 'inria-3dgs', '--seed', "$seme")) }
}

# 7-9. Controllo delle immagini di riferimento, report e materiali dell'articolo.
if (Passo) {
    Esegui 'Controllo delle immagini di riferimento' @('-m', 'app.check_gt', '--project', $Progetto)
    foreach ($fattore in 2, 4) {  # anche alle risoluzioni ridotte delle prove di sensibilita'
        Esegui "Controllo delle immagini di riferimento, riduzione $fattore" @('-m', 'app.check_gt', '--project', $Progetto, '--downscale', "$fattore")
    }
}
if (Passo) { Esegui 'Report' @('-m', 'app.report', '--project', $Progetto) }
if (Passo) { Esegui 'Materiali per l''articolo' @('-m', 'app.paper', '--project', $Progetto, '--out', "paper\$Progetto", '--scartate', 'foto\LEGGIMI.txt') }
Add-Content -Path $registro -Encoding utf8 -Value "`n### tutto finito $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
