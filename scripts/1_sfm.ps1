# Passo 1 - Structure-from-Motion: foto -> pose delle camere + nuvola sparsa (COLMAP),
# poi conversione nel formato nerfstudio (transforms.json + immagini ridimensionate).
# Foto attese in data\<Scene>\input. Uso:  .\scripts\1_sfm.ps1 -Scene arco
param(
    [string]$Scene = 'arco',
    # exhaustive: tutte le coppie (foto sparse, fino a qualche centinaio); sequential: frame di un video
    [ValidateSet('exhaustive', 'sequential')][string]$Matcher = 'exhaustive',
    # single: una sola camera per tutte le foto; per_folder: una camera per sottocartella di input;
    # per_image: parametri interni stimati per ogni immagine (camere/zoom diversi)
    [ValidateSet('single', 'per_folder', 'per_image')][string]$Camera = 'single'
)
. "$PSScriptRoot\..\activate.ps1"
$scenePath = "$PSScriptRoot\..\data\$Scene"
$images = "$scenePath\input"
$colmap = "$scenePath\colmap"
if (-not (Get-ChildItem $images -File -Recurse -ErrorAction SilentlyContinue)) { throw "Nessuna foto in $images" }
New-Item -ItemType Directory -Force "$colmap\sparse" | Out-Null

function Step($name) { if ($LASTEXITCODE -ne 0) { throw "$name fallito (exit $LASTEXITCODE)" } }

# Non si usa la chiamata a COLMAP integrata in ns-process-data: nerfstudio 1.1.5 passa
# opzioni (--SiftExtraction.use_gpu) che in COLMAP 4.x non esistono piu'.
$cameraArgs = switch ($Camera) {
    'single' { '--ImageReader.single_camera', '1' }
    'per_folder' { '--ImageReader.single_camera_per_folder', '1' }
    'per_image' { '--ImageReader.single_camera_per_image', '1' }
}
colmap feature_extractor --database_path "$colmap\database.db" --image_path $images `
    --ImageReader.camera_model OPENCV @cameraArgs --FeatureExtraction.use_gpu 1
Step 'feature_extractor'

colmap "${Matcher}_matcher" --database_path "$colmap\database.db" --FeatureMatching.use_gpu 1
Step 'matcher'

colmap mapper --database_path "$colmap\database.db" --image_path $images --output_path "$colmap\sparse"
Step 'mapper'

$models = @(Get-ChildItem "$colmap\sparse" -Directory)
if ($models.Count -gt 1) {
    Write-Warning "COLMAP ha prodotto $($models.Count) modelli separati: le foto non si collegano tutte. Viene usato sparse\0."
}
colmap model_analyzer --path "$colmap\sparse\0"

ns-process-data images --data $images --output-dir $scenePath --skip-colmap
Step 'ns-process-data'
