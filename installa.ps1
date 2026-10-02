# Installa tutto il necessario nella cartella del programma: Python, ambiente con PyTorch e
# nerfstudio, moduli CUDA gia' compilati, COLMAP, FFmpeg, MeshLab e il codice Inria.
# Non richiede diritti di amministratore. Puo' essere rilanciato: salta cio' che e' gia' presente.
# Di norma lo avvia il programma di installazione (Setup); a mano: doppio clic su Installa.bat
param(
    [switch]$SenzaCollegamento,  # non creare il collegamento sul desktop (lo crea il programma di installazione)
    [switch]$SoloControlli,      # verifica solo che il computer e la cartella siano adatti, senza installare
    [string]$Cartella            # cartella da verificare con -SoloControlli (predefinita: quella dello script)
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = $PSScriptRoot
if ($SoloControlli -and $Cartella) { $root = $Cartella.TrimEnd('') } else { Set-Location $root }

$repo = 'Mr-Flower/studio-gaussian-splatting-laboratorio-ictbc'
# Release a cui sono allegati i moduli compilati: non cambiano a ogni versione del programma.
$releaseModuli = 'v0.2.0'
$pythonUrl = 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe'
$colmapUrl = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip'
$ffmpegUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$meshlabUrl = 'https://github.com/cnr-isti-vclab/meshlab/releases/download/MeshLab-2025.07/MeshLab2025.07-windows_x86_64.zip'
$inriaCommit = '54c035f7834b564019656c3e3fcc3646292f727d'
$inriaUrl = "https://github.com/graphdeco-inria/gaussian-splatting/archive/$inriaCommit.zip"
# Moduli CUDA compilati per Python 3.10, PyTorch 2.4 e CUDA 12.4, allegati alla release.
$wheels = @(
    'tinycudann-2.0-cp310-cp310-win_amd64.whl',
    'diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl',
    'simple_knn-0.0.0-cp310-cp310-win_amd64.whl',
    'fused_ssim-0.0.0-cp310-cp310-win_amd64.whl'
)
$driverMinimo = 551.61  # primo driver NVIDIA che supporta CUDA 12.4

$passi = 6
$passo = 0
# Il programma di installazione legge queste righe per mostrare l'avanzamento.
function Passo($testo) { $script:passo++; Write-Host "`n== [$passo/$passi] $testo" -ForegroundColor Cyan }
function Avviso($testo) { Write-Host "ATTENZIONE: $testo" -ForegroundColor Yellow }

function Scarica($url, $destinazione) {
    Write-Host "   scarico $url"
    Invoke-WebRequest -Uri $url -OutFile $destinazione -UseBasicParsing
}

function Scompatta($url, $cartella) {
    # Scarica uno zip e lo scompatta in tools\<cartella>.
    $zip = Join-Path $env:TEMP ("sgs_" + [IO.Path]::GetRandomFileName() + ".zip")
    Scarica $url $zip
    Expand-Archive -Path $zip -DestinationPath (Join-Path $root "tools\$cartella") -Force
    Remove-Item $zip -Confirm:$false
}

function Esegui($programma, $argomenti, $errore) {
    & $programma @argomenti
    if ($LASTEXITCODE -ne 0) { throw $errore }
}

try {
    Passo "Controlli"
    if (-not [Environment]::Is64BitOperatingSystem) { throw "Serve Windows a 64 bit." }
    # Windows, di norma, si ferma a 260 caratteri e i pacchetti Python installati hanno percorsi
    # interni lunghi fino a 175 caratteri: oltre i 75 della cartella l'installazione fallirebbe.
    if ($root.Length -gt 75) {
        throw "Il percorso della cartella del programma e' troppo lungo ($($root.Length) caratteri, massimo 75): sceglierne uno piu' breve, ad esempio C:\GaussianSplatting."
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "Scheda NVIDIA non trovata (manca nvidia-smi). Il programma richiede una scheda NVIDIA con i suoi driver."
    }
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1) -split ',\s*'
    Write-Host "   scheda: $($gpu[0]), driver $($gpu[1])"
    if ([double]::Parse($gpu[1], [Globalization.CultureInfo]::InvariantCulture) -lt $driverMinimo) {
        throw "Il driver NVIDIA e' troppo vecchio: serve la versione $driverMinimo o successiva. Aggiornarlo da nvidia.com e rilanciare."
    }
    $liberi = [math]::Round((Get-PSDrive ($root.Substring(0, 1))).Free / 1GB)
    Write-Host "   spazio libero: $liberi GB"
    if ($liberi -lt 25) { throw "Servono almeno 25 GB liberi sul disco del programma (liberi: $liberi GB)." }
    if ($SoloControlli) { exit 0 }

    Passo "Python 3.10"
    $python = @(
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:ProgramFiles\Python310\python.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $python) {
        $setup = Join-Path $env:TEMP 'python-3.10.11-amd64.exe'
        Scarica $pythonUrl $setup
        Write-Host "   installo Python per l'utente corrente"
        $p = Start-Process $setup -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0', 'Include_test=0' -Wait -PassThru
        if ($p.ExitCode -ne 0) { throw "Installazione di Python non riuscita (codice $($p.ExitCode))." }
        $python = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    }
    Write-Host "   $python"

    Passo "Ambiente Python (PyTorch, nerfstudio, gsplat: alcuni GB, puo' richiedere parecchi minuti)"
    $py = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { Esegui $python @('-m', 'venv', (Join-Path $root '.venv')) "Creazione dell'ambiente non riuscita." }
    Esegui $py @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip', 'wheel', 'setuptools==69.5.1') "Aggiornamento di pip non riuscito."
    Esegui $py @('-m', 'pip', 'install', '--quiet', '-r', (Join-Path $root 'requirements.txt')) "Installazione dei pacchetti Python non riuscita."

    Passo "Moduli CUDA gia' compilati"
    $cartellaWheel = Join-Path $root 'tools\wheels'
    New-Item -ItemType Directory -Force $cartellaWheel | Out-Null
    foreach ($nome in $wheels) {
        $file = Join-Path $cartellaWheel $nome
        try {
            if (-not (Test-Path $file)) { Scarica "https://github.com/$repo/releases/download/$releaseModuli/$nome" $file }
            Esegui $py @('-m', 'pip', 'install', '--quiet', '--no-deps', $file) "installazione non riuscita"
        } catch {
            Remove-Item $file -ErrorAction SilentlyContinue -Confirm:$false
            if ($nome -like 'tinycudann*') { Avviso "tiny-cuda-nn non disponibile: i metodi NeRF funzioneranno, ma molto piu' lentamente." }
            else { Avviso "$nome non disponibile: il metodo 'Gaussian splatting originale (Inria)' non sara' utilizzabile." }
        }
    }

    Passo "COLMAP, FFmpeg, MeshLab e codice Inria"
    if (-not (Test-Path "$root\tools\colmap-4.2.1\bin\colmap.exe")) { Scompatta $colmapUrl 'colmap-4.2.1' }
    if (-not (Get-ChildItem "$root\tools" -Filter 'ffmpeg-*' -Directory -ErrorAction SilentlyContinue)) { Scompatta $ffmpegUrl '.' }
    if (-not (Test-Path "$root\tools\meshlab\meshlab.exe")) { Scompatta $meshlabUrl 'meshlab' }
    if (-not (Test-Path "$root\tools\gaussian-splatting\train.py")) {
        Scompatta $inriaUrl '.'
        Rename-Item "$root\tools\gaussian-splatting-$inriaCommit" 'gaussian-splatting'
    }

    Passo "Verifica"
    Esegui $py @('-c', "import torch; assert torch.cuda.is_available(), 'PyTorch non vede la scheda grafica'; print('   PyTorch', torch.__version__, 'su', torch.cuda.get_device_name(0))") "PyTorch non riesce a usare la scheda grafica."
    Esegui $py @('-c', "from app import config; m = config.missing_components(); assert not m, m; print('   metodi disponibili:', ', '.join(config.available_methods()))") "Mancano dei componenti."

    if (-not $SenzaCollegamento) {
        $collegamento = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Studio Gaussian Splatting.lnk'))
        $collegamento.TargetPath = Join-Path $root 'Avvia.bat'
        $collegamento.WorkingDirectory = $root
        $collegamento.WindowStyle = 7
        $collegamento.Save()
        Write-Host "   collegamento creato sul desktop"
    }

    Write-Host "`nInstallazione completata. Avviare il programma con Avvia.bat o dal collegamento sul desktop." -ForegroundColor Green
    exit 0
} catch {
    Write-Host "`nERRORE: $($_.Exception.Message)" -ForegroundColor Red
    if (-not $SoloControlli) { Write-Host "L'installazione puo' essere rilanciata: riprende da dove si e' fermata." }
    exit 1
}
