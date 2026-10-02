# Studio Gaussian Splatting — Laboratorio ICTBC

Strumento di ricerca per confrontare, sullo stesso set di fotografie, metodi diversi di
ricostruzione 3D: **3D Gaussian Splatting**, **NeRF** e **fotogrammetria classica** (multi-view
stereo). Tutti i metodi partono dallo stesso allineamento delle camere, così le differenze nei
risultati dipendono dal metodo e non dai dati di partenza.

Il caso di studio è il rilievo fotografico da drone dell'Arco di Traiano a Benevento, ma il
programma funziona con qualsiasi cartella di foto.

Uso previsto: ricerca scientifica, non commerciale.

## Metodi a confronto

| Metodo | Famiglia | Rappresentazione | Implementazione | Risultato esportato |
|---|---|---|---|---|
| `splatfacto` | 3D Gaussian Splatting [1] | gaussiane 3D esplicite, rasterizzate | nerfstudio [3] + gsplat [2] | `splat.ply` |
| `splatfacto-big` | 3D Gaussian Splatting [1] | come sopra, con più gaussiane | nerfstudio + gsplat | `splat.ply` |
| `nerfacto` | NeRF [4] | campo di radianza neurale, rendering volumetrico | nerfstudio [3] | nuvola di punti `point_cloud.ply` |
| `nerfacto-big` | NeRF [4] | come sopra, rete più grande | nerfstudio | nuvola di punti `point_cloud.ply` |
| Fotogrammetria | Multi-view stereo [6] + Poisson [7] | nuvola densa e mesh | COLMAP | `fused.ply`, `mesh-poisson.ply` |

L'allineamento delle camere (structure-from-motion [5]) è fatto una sola volta con COLMAP ed è
condiviso da tutti i metodi.

Sono elencati solo i metodi verificati su questa installazione. I metodi NeRF girano con
l'implementazione PyTorch delle codifiche, perché `tiny-cuda-nn` non è installato: i risultati sono
validi, i tempi di calcolo sono molto più lunghi di quelli ottenibili con `tiny-cuda-nn` e vanno
letti di conseguenza.

## Cosa viene misurato

Per ogni run di training il programma registra:

- **Qualità delle immagini sintetizzate**: PSNR, SSIM [8] e LPIPS [9] sulle viste di test. Il 10%
  delle foto, scelte a intervalli regolari, è escluso dal training e usato solo per la valutazione
  (impostazione predefinita di nerfstudio); la suddivisione è la stessa per tutti i metodi.
- **Tempi**: durata del training (dell'intero comando, caricamento dei dati incluso), della
  valutazione e dell'esportazione.
- **Dimensioni**: numero di gaussiane o di punti esportati, dimensione del modello salvato.
- **Velocità di rendering** in fotogrammi al secondo, durante la valutazione.
- **Condizioni dell'esperimento**: iterazioni, risoluzione delle immagini, numero di foto, scheda
  grafica, versioni di nerfstudio, gsplat, PyTorch e COLMAP.

Per la fotogrammetria classica sono registrati i tempi di ogni fase, il numero di punti della
nuvola densa e di vertici della mesh.

I risultati sono raccolti in una tabella di confronto, esportabile in CSV.

### Limiti del confronto

- Le metriche misurano la fedeltà delle immagini, non l'accuratezza geometrica. La mesh
  fotogrammetrica non produce immagini sintetizzate e non ha quindi PSNR, SSIM o LPIPS: il confronto
  con gli altri metodi su questo piano richiede un riferimento metrico indipendente.
- Sono confrontabili solo i run fatti sullo stesso allineamento e alla stessa risoluzione. Se
  l'allineamento viene rifatto, i run precedenti sono segnalati come non più confrontabili.
- Nessuno dei metodi configurati compensa le differenze di esposizione tra le foto.
- Ogni configurazione è eseguita una volta: le metriche non hanno una stima della variabilità tra
  esecuzioni ripetute.

## Caso di studio: Arco di Traiano

Rilievo del 19 luglio 2024 con drone DJI Mavic 3 (camera Hasselblad L2D-20c, 5280×3956 pixel): 906
fotografie scattate tra le 9:25 e le 16:08, con esposizione e bilanciamento del bianco automatici.
853 sono i JPG della camera; 53 fotogrammi, disponibili solo in formato DNG, sono stati sviluppati
con `scripts/0_prepare_arco.py`.

Le fotografie e i modelli non sono nel repository per le loro dimensioni. I risultati del confronto
saranno aggiunti qui al termine delle elaborazioni.

## Requisiti

- Windows 10/11 a 64 bit
- Scheda NVIDIA con driver recenti (sviluppato su RTX A6000, 48 GB)
- Python 3.10

## Installazione

Dalla cartella del repository, in PowerShell:

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
| `tools\meshlab\` | MeshLab portabile (facoltativo, per aprire la mesh) | https://github.com/cnr-isti-vclab/meshlab/releases |

All'avvio il programma segnala i componenti mancanti.

## Uso con l'interfaccia grafica

Doppio clic su `Avvia.bat`.

1. Scegli la cartella delle foto: il nome del progetto viene proposto automaticamente.
2. Seleziona i metodi da confrontare e i passi da eseguire.
3. Premi «Avvia».

I passi sono cinque:

| Passo | Cosa fa |
|---|---|
| Allineamento | Posizione e parametri delle camere con COLMAP. Si fa una volta per progetto. |
| Training | Allena ogni metodo selezionato. Ogni avvio crea un nuovo run, senza sovrascrivere i precedenti. |
| Valutazione | Calcola PSNR, SSIM e LPIPS sulle viste di test. |
| Esportazione | Salva il modello di ogni run in `exports\`. |
| Mesh | Nuvola densa e mesh con COLMAP. Con centinaia di foto richiede molte ore. |

La scheda «Confronto dei metodi» elenca i run del progetto con le loro misure. Da lì si apre un
modello nel viewer di nerfstudio, si apre un gaussian splat in [SuperSplat](https://superspl.at/editor)
per pulirlo, si apre la mesh in MeshLab e si esporta la tabella in CSV.

## Uso da riga di comando

La stessa pipeline è disponibile senza interfaccia, per esecuzioni in serie:

```powershell
# allineamento, training, valutazione ed esportazione di due metodi
.venv\Scripts\python -m app.cli run --project arco --photos D:\foto\arco --methods splatfacto nerfacto

# solo alcuni passi, su un progetto già allineato
.venv\Scripts\python -m app.cli run --project arco --steps train eval export --methods splatfacto-big --iterations 30000

# tabella di confronto, anche in CSV
.venv\Scripts\python -m app.cli report --project arco --csv confronto_arco.csv
```

`python -m app.cli run --help` elenca tutte le opzioni.

## Dove finiscono i risultati

```
data\<progetto>\                      allineamento, comune a tutti i metodi
    project.json                      impostazioni del progetto
    alignment.json                    foto allineate, punti, errore di riproiezione
    colmap\                           ricostruzione sparsa; colmap\dense\ per nuvola densa e mesh
    pipeline.log                      log completo delle elaborazioni
outputs\<progetto>\<metodo>\<data>\   un run di training
    run.json                          parametri, tempi, versioni
    metrics.json                      PSNR, SSIM, LPIPS
exports\<progetto>\<metodo>_<data>\   modello esportato
```

## Sviluppo

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests
```

I test coprono la lettura dei log, la costruzione dei comandi, il registro dei run e i controlli
prima dell'avvio; non richiedono la scheda grafica. `docs\audit.md` descrive la revisione che ha
portato alla versione 0.1.0 e i punti ancora aperti.

## Software di terze parti

Questo repository contiene solo il codice che coordina strumenti esistenti; non ne ridistribuisce
nessuno.

| Software | Uso | Licenza |
|---|---|---|
| [COLMAP](https://colmap.github.io/) | allineamento, multi-view stereo, mesh | BSD 3-Clause |
| [nerfstudio](https://docs.nerf.studio/) | training, valutazione, esportazione, viewer | Apache 2.0 |
| [gsplat](https://docs.gsplat.studio/) | rasterizzazione delle gaussiane | Apache 2.0 |
| [PyTorch](https://pytorch.org/) | calcolo su GPU | BSD 3-Clause |
| [PySide6](https://doc.qt.io/qtforpython/) (Qt) | interfaccia grafica | LGPL v3 |
| [FFmpeg](https://ffmpeg.org/) | ridimensionamento delle immagini | GPL v3 (build usata) |
| [SuperSplat](https://github.com/playcanvas/supersplat) | pulizia dei gaussian splat, nel browser | MIT |
| [MeshLab](https://www.meshlab.net/) | visualizzazione della mesh | GPL v3 |

### Rapporto con l'implementazione originale di 3D Gaussian Splatting

Il metodo è quello di Kerbl et al. [1], la cui implementazione di riferimento è
[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).
Quel codice è distribuito da Inria e Max Planck Institut für Informatik con una licenza che ne
consente l'uso solo per ricerca e valutazione, non commerciale, e chiede di citare la pubblicazione.
L'uso che se ne fa in questo studio, ricerca accademica non commerciale, rientra in quei termini.

Questo repository non contiene né esegue codice di quella implementazione: i gaussian splat sono
allenati con gsplat, una reimplementazione indipendente con licenza Apache 2.0. La pubblicazione
originale va comunque citata in ogni lavoro che usi questi risultati:

```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Riferimenti

1. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics 42(4), 2023.
2. V. Ye et al. *gsplat: An Open-Source Library for Gaussian Splatting*. arXiv:2409.06765, 2024.
3. M. Tancik et al. *Nerfstudio: A Modular Framework for Neural Radiance Field Development*. ACM SIGGRAPH 2023.
4. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV 2020.
5. J. L. Schönberger, J.-M. Frahm. *Structure-from-Motion Revisited*. CVPR 2016.
6. J. L. Schönberger, E. Zheng, M. Pollefeys, J.-M. Frahm. *Pixelwise View Selection for Unstructured Multi-View Stereo*. ECCV 2016.
7. M. Kazhdan, H. Hoppe. *Screened Poisson Surface Reconstruction*. ACM Transactions on Graphics 32(3), 2013.
8. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing 13(4), 2004.
9. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018.

## Licenza

Per questo repository non è ancora stata scelta una licenza.
