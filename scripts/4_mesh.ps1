# Passo 4 - Mesh con la ricostruzione densa di COLMAP (multi-view stereo su GPU + Poisson).
# Richiede il passo 1. Risultato: data\<Scene>\colmap\dense\mesh-poisson.ply (colori per vertice),
# apribile in MeshLab / CloudCompare / Blender.
# Uso:  .\scripts\4_mesh.ps1 -Scene arco
param(
    [string]$Scene = 'arco',
    # Lato massimo delle immagini usate per le mappe di profondita': piu' alto = piu' dettaglio e piu' tempo
    [int]$MaxImageSize = 2400
)
. "$PSScriptRoot\..\activate.ps1"
$scenePath = "$PSScriptRoot\..\data\$Scene"
$dense = "$scenePath\colmap\dense"

function Step($name) { if ($LASTEXITCODE -ne 0) { throw "$name fallito (exit $LASTEXITCODE)" } }

colmap image_undistorter --image_path "$scenePath\input" --input_path "$scenePath\colmap\sparse\0" `
    --output_path $dense --output_type COLMAP --max_image_size $MaxImageSize
Step 'image_undistorter'

colmap patch_match_stereo --workspace_path $dense --PatchMatchStereo.geom_consistency 1
Step 'patch_match_stereo'

colmap stereo_fusion --workspace_path $dense --output_path "$dense\fused.ply"
Step 'stereo_fusion'

colmap poisson_mesher --input_path "$dense\fused.ply" --output_path "$dense\mesh-poisson.ply"
Step 'poisson_mesher'
