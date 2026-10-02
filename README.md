# Gaussian splatting dell'Arco di Traiano (Benevento)

Pipeline per ottenere un modello 3D a gaussian splatting, e una mesh, da un set di fotografie.
È nata per il rilievo da drone dell'Arco di Traiano a Benevento, ma funziona con qualsiasi cartella di foto.

Usa solo strumenti liberi: [COLMAP](https://colmap.github.io/) per l'allineamento delle foto,
[nerfstudio](https://docs.nerf.studio/) con [gsplat](https://docs.gsplat.studio/) per il training,
[SuperSplat](https://superspl.at/editor) per pulire il modello e [MeshLab](https://www.meshlab.net/) per la mesh.

Le foto e i risultati non sono nel repository: sono troppo grandi.

## Requisiti

- Windows 10/11 a 64 bit
- Scheda NVIDIA con driver recenti (provato su RTX A6000)
- Python 3.10

## Installazione

Dalla cartella del progetto, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Poi vanno scaricati e scompattati in `tools\` (non serve installarli):

| Cartella | Cosa | Da dove |
|---|---|---|
| `tools\colmap-4.2.1\` | COLMAP 4.2.1, build Windows con CUDA | https://github.com/colmap/colmap/releases |
| `tools\ffmpeg-*\` | FFmpeg, build "essentials" | https://www.gyan.dev/ffmpeg/builds/ |
| `tools\git\` | MinGit | https://github.com/git-for-windows/git/releases |
| `tools\meshlab\` | MeshLab portabile (facoltativo, per aprire la mesh) | https://github.com/cnr-isti-vclab/meshlab/releases |

## Uso con l'interfaccia grafica

Doppio clic su `Avvia.bat`. Si sceglie la cartella delle foto, si spuntano i passi da eseguire e si preme "Avvia".
I passi sono quattro:

1. **Allineamento** delle foto con COLMAP (posizione delle camere e nuvola di punti sparsa)
2. **Training** del gaussian splat
3. **Esportazione** in `exports\<progetto>\splat.ply`, da aprire in SuperSplat
4. **Mesh** con la ricostruzione densa di COLMAP, in `data\<progetto>\colmap\dense\mesh-poisson.ply`

La mesh è molto più lenta degli altri passi: con centinaia di foto servono molte ore.

## Uso da riga di comando

Gli stessi passi sono disponibili come script PowerShell, con le foto in `data\<scena>\input`:

```powershell
.\scripts\1_sfm.ps1 -Scene arco -Camera per_folder
.\scripts\2_train.ps1 -Scene arco
.\scripts\3_export.ps1 -Scene arco
.\scripts\4_mesh.ps1 -Scene arco
```

`scripts\0_prepare_arco.py` è specifico per il rilievo dell'arco: usa i JPG della camera e sviluppa
i fotogrammi presenti solo in formato DNG.

## Note

- nerfstudio 1.1.5 non è compatibile con le opzioni di COLMAP 4.x, quindi l'allineamento viene
  lanciato direttamente e poi convertito con `ns-process-data --skip-colmap`.
- `setuptools` è fissato alla 69.5.1 perché PyTorch 2.1 richiede `pkg_resources`.
