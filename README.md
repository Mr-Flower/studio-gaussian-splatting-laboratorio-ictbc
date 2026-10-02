# Studio Gaussian Splatting — Laboratorio ICTBC

**Confronto riproducibile tra 3D Gaussian Splatting, NeRF e fotogrammetria multi-vista per la
ricostruzione 3D di beni architettonici da fotografie: metodo, strumento software e caso di studio
dell'Arco di Traiano a Benevento.**

## Abstract

**Contesto.** La ricostruzione tridimensionale da fotografie dispone oggi di tre famiglie di metodi:
la fotogrammetria multi-vista, che produce nuvole dense e mesh; i campi di radianza neurali (NeRF),
che codificano la scena in una rete neurale; il 3D Gaussian Splatting, che la rappresenta con
gaussiane esplicite rasterizzabili in tempo reale. I confronti pubblicati usano spesso dati di
partenza, viste di test, risoluzioni e codici di valutazione diversi da metodo a metodo, e questo
rende difficile attribuire le differenze osservate al metodo anziché al protocollo.

**Obiettivo.** Mettere a confronto i tre approcci a parità di condizioni su un rilievo reale di un
bene architettonico, e rendere il confronto ripetibile da chiunque su un proprio set di fotografie.

**Metodi.** Si presenta uno strumento open source che, data una cartella di fotografie, (i) ne
analizza la qualità e propone l'esclusione di quelle poco nitide, male esposte o quasi duplicate;
(ii) calcola un unico allineamento delle camere con COLMAP; (iii) allena sullo stesso allineamento
cinque configurazioni — 3D Gaussian Splatting nell'implementazione originale di Inria e in quella
di gsplat (due varianti), NeRF con nerfacto (due varianti) — e ricostruisce nuvola densa e mesh con
la fotogrammetria multi-vista; (iv) valuta ogni modello con un protocollo unico: stesse fotografie
di training e di test (una su otto esclusa dal training), stessa risoluzione, stesso codice per
PSNR, SSIM e LPIPS, stessa misura del tempo di rendering; (v) registra tempi di calcolo,
dimensioni dei modelli e versioni del software, e produce un report con grafici e viste a
confronto. Il caso di studio è l'Arco di Traiano a Benevento, rilevato da drone con 906 fotografie,
887 delle quali selezionate dall'analisi automatica.

**Risultati.** L'intera procedura è stata verificata su un set di prova di 40 fotografie con tutte
le configurazioni. Il confronto completo sul caso di studio è in corso di elaborazione: i valori
numerici saranno riportati in questa sezione e nella sezione [Risultati](#risultati) al suo termine.

**Conclusioni.** Lo strumento rende il confronto tra metodi un'operazione di routine: l'utente
fornisce solo la cartella delle fotografie e ottiene i modelli, le misure e il report. Le
conclusioni sul caso di studio saranno formulate sui risultati misurati.

**Parole chiave:** 3D Gaussian Splatting; Neural Radiance Fields; fotogrammetria; structure from
motion; multi-view stereo; beni culturali; rilievo da drone; riproducibilità.

### Abstract (English)

*Background.* Image-based 3D reconstruction now offers three families of methods: multi-view
photogrammetry, neural radiance fields (NeRF) and 3D Gaussian Splatting. Published comparisons
often differ in input data, held-out views, image resolution and evaluation code from one method to
the next, which makes it hard to attribute observed differences to the method rather than to the
protocol. *Objective.* To compare the three approaches under identical conditions on a real survey
of an architectural heritage asset, and to make the comparison repeatable on any photo set.
*Methods.* We present an open-source tool that, given a folder of photographs, screens them for
blur, exposure and near-duplicates; computes a single camera alignment with COLMAP; trains five
configurations on that alignment — 3D Gaussian Splatting in the original Inria implementation and
in gsplat (two variants), and NeRF with nerfacto (two variants) — and reconstructs a dense cloud
and mesh by multi-view stereo; evaluates every model with one protocol (same training and test
images, one in eight held out; same resolution; same code for PSNR, SSIM and LPIPS; same rendering
time measurement); and records computing times, model sizes and software versions, producing a
report with charts and side-by-side views. The case study is the Arch of Trajan in Benevento
(Italy), surveyed by drone with 906 photographs, 887 of which were retained by the automatic
screening. *Results.* The whole procedure has been verified on a 40-image test set with all
configurations; the full comparison on the case study is in progress and its figures will be
reported here. *Keywords:* 3D Gaussian Splatting; Neural Radiance Fields; photogrammetry; structure
from motion; multi-view stereo; cultural heritage; UAV survey; reproducibility.

## Metodi a confronto

| Metodo | Famiglia | Rappresentazione | Implementazione | Risultato esportato |
|---|---|---|---|---|
| `inria-3dgs` | 3D Gaussian Splatting [1] | gaussiane 3D esplicite, rasterizzate | codice originale Inria [1] | `splat.ply` |
| `splatfacto` | 3D Gaussian Splatting [1] | come sopra | nerfstudio [3] + gsplat [2] | `splat.ply` |
| `splatfacto-big` | 3D Gaussian Splatting [1] | come sopra, con più gaussiane | nerfstudio + gsplat | `splat.ply` |
| `nerfacto` | NeRF [4] | campo di radianza neurale, rendering volumetrico | nerfstudio [3] | nuvola di punti `point_cloud.ply` |
| `nerfacto-big` | NeRF [4] | come sopra, rete più grande | nerfstudio | nuvola di punti `point_cloud.ply` |
| Fotogrammetria | Multi-view stereo [6] + Poisson [7] | nuvola densa e mesh | COLMAP | `fused.ply`, `mesh-poisson.ply` |

Ogni metodo è usato con i parametri predefiniti della sua implementazione; il numero di iterazioni
è lo stesso per tutti.

## Protocollo sperimentale

1. **Selezione delle fotografie.** Per ogni foto si misurano la nitidezza (rapporto tra dettaglio
   fine e dettaglio grossolano, calcolato sulle sole zone ricche di dettaglio), la frazione di pixel
   bruciati o neri e un'impronta percettiva. Si propone di escludere le foto con nitidezza sotto il
   40% della mediana della propria cartella, quelle con più di metà dei pixel bruciati o neri e i
   quasi-doppioni (si tiene la più nitida). La decisione resta all'utente; nessun file viene toccato.
2. **Allineamento.** Structure-from-motion con COLMAP [5], una sola volta, condiviso da tutti i metodi.
3. **Suddivisione training/test.** Le foto allineate, ordinate per nome, vanno al test una ogni
   otto (convenzione di Mip-NeRF 360 e di 3D Gaussian Splatting). L'elenco è scritto nei formati
   letti da nerfstudio e dal codice Inria, e in fase di valutazione si verifica che ogni metodo sia
   stato valutato esattamente su quelle foto.
4. **Risoluzione.** Lo stesso fattore di riduzione (1, 2, 4 o 8) per tutti i metodi allenati
   insieme: il più piccolo con cui le foto entrano in memoria per ognuno di essi, contando al più
   metà della memoria centrale e della scheda grafica. I livelli di qualità inferiori aggiungono un
   tetto al lato maggiore (1600 o 800 pixel) e riducono le iterazioni. Nel report entrano solo run
   fatti alla stessa risoluzione e con lo stesso numero di iterazioni.
5. **Valutazione.** Per ogni vista di test si genera l'immagine con il modello allenato e si
   calcolano PSNR, SSIM [8] e LPIPS [9] (rete AlexNet) con lo stesso codice per tutti i metodi. La
   velocità di rendering è il tempo medio di generazione di una vista, esclusa la prima.
6. **Tracciabilità.** Ogni run salva parametri, durata di ogni passo, scheda grafica, versioni dei
   componenti e l'impronta dell'allineamento usato.

### Limiti

- Le metriche misurano la fedeltà delle immagini, non l'accuratezza geometrica. La fotogrammetria
  non produce immagini da nuovi punti di vista e non ha quindi PSNR, SSIM o LPIPS: il confronto con
  gli altri metodi sul piano geometrico richiede un riferimento metrico indipendente.
- L'immagine vera con cui si confronta ogni vista è preparata dal programma che ha allenato il
  modello: nerfstudio e il codice Inria correggono la distorsione dell'obiettivo con procedure
  diverse, quindi le immagini di riferimento coincidono nel contenuto ma non pixel per pixel.
- Nessuno dei metodi configurati compensa le differenze di esposizione tra le foto.
- Ogni configurazione è eseguita una volta: la variabilità tra esecuzioni ripetute non è stimata.
- Sono confrontabili solo i run fatti sullo stesso allineamento; se viene rifatto, i run precedenti
  sono segnalati ed esclusi dal report.

## Caso di studio: Arco di Traiano

Rilievo del 19 luglio 2024 con drone DJI Mavic 3 (camera Hasselblad L2D-20c, 5280×3956 pixel): 906
fotografie scattate tra le 9:25 e le 16:08, con esposizione e bilanciamento del bianco automatici.
853 sono i JPG della camera; 53 fotogrammi, disponibili solo in formato DNG, sono stati sviluppati
con `scripts/0_prepare_arco.py` e sono trattati come una seconda camera. L'analisi automatica ha
scartato 19 foto (9 poco nitide, 11 quasi-doppioni, una in entrambe le categorie): il set usato è
di 887 fotografie.

Le fotografie e i modelli non sono nel repository per le loro dimensioni.

## Risultati

Il confronto sul caso di studio è in corso. Al termine, questa sezione riporterà la tabella delle
misure e i grafici prodotti dal report (`reports/<progetto>/`).

## Installazione

Requisiti: Windows 10/11 a 64 bit; scheda NVIDIA RTX (serie 20, 30, 40 o professionali equivalenti)
con driver 551.61 o successivo; circa 25 GB liberi. Non servono diritti di amministratore.

1. Scaricare `GaussianSplatting-Setup-<versione>.exe` dalla [release](../../releases) e avviarlo.
2. Seguire la procedura guidata: licenza, cartella del programma (proposta:
   `Programmi\Gaussian Splatting` dell'utente), cartella di lavoro per progetti e risultati, icona
   sul desktop.

La procedura copia il programma e poi scarica e configura tutto il necessario, mostrando
l'avanzamento: Python 3.10, PyTorch, nerfstudio e gsplat, COLMAP, FFmpeg, MeshLab e il codice Inria
(circa 6 GB, da 10 a 40 minuti). I moduli CUDA già compilati (tiny-cuda-nn e i moduli del codice
Inria) sono dentro il Setup. Se lo scaricamento si interrompe, «Completa o ripara l'installazione»
nel menu Start riprende da dove si era fermato. Il programma si disinstalla da «App installate» di
Windows; la cartella di lavoro non viene toccata.

Il Setup non è firmato: Windows può mostrare l'avviso «PC protetto da Windows» («Ulteriori
informazioni», poi «Esegui comunque»). La cartella del programma non deve superare i 75 caratteri
di percorso, per il limite di Windows sui percorsi lunghi: la procedura lo verifica.

In alternativa, dal codice sorgente: scompattare il repository in una cartella dal percorso breve e
fare doppio clic su `Installa.bat`; il programma si avvia poi con `Avvia.bat` e tiene progetti e
risultati nella propria cartella.

## Uso

Si avvia dall'icona sul desktop o dal menu Start. Basta indicare la cartella
delle foto, scegliere il metodo dal menu e premere «Crea il modello 3D»: allineamento, risoluzione,
valutazione ed esportazione sono impostati dal programma per la resa migliore che il computer regge.

| Scheda | Cosa si fa |
|---|---|
| **1. Foto** | Si analizzano le foto e si decide quali escludere dall'allineamento. |
| **2. Crea il modello** | Si sceglie il metodo dal menu; la qualità parte da «Massima» e si può abbassare per fare prima. «Crea il modello 3D» fa tutto il resto. «Test: confronta più metodi…» fa spuntare i metodi da confrontare, li allena nelle stesse condizioni e apre il report. Durante il training dei metodi di nerfstudio il risultato si può guardare nel browser. |
| **3. Risultati e confronto** | Tabella dei run con le loro misure. Da qui si apre un modello nel viewer, lo si apre in [SuperSplat](https://superspl.at/editor) per pulirlo e pubblicarlo, si apre la mesh in MeshLab, si genera il report e si esporta la tabella in CSV. |

| Qualità | Iterazioni | Risoluzione delle foto |
|---|---|---|
| Massima (predefinita) | 30.000 | la più alta che entra in memoria |
| Alta | 30.000 | lato maggiore fino a 1600 px |
| Media | 15.000 | lato maggiore fino a 1600 px |
| Bozza veloce | 7.000 | lato maggiore fino a 800 px |

Con più metodi insieme la risoluzione è quella che regge anche il più esigente: il codice Inria e i
NeRF tengono le foto in memoria in virgola mobile e a parità di computer arrivano a risoluzioni più
basse di gsplat. In «Opzioni avanzate» si trovano il modello di camera, il tipo di matching, la
ripetizione dell'allineamento e la fotogrammetria classica (mesh), molto più lenta degli altri passi:
con centinaia di foto servono molte ore.

### Riga di comando

```powershell
# confronto completo: tutti i metodi, valutazione, esportazione e report
.venv\Scripts\python -m app.cli test --project arco --photos D:\foto\arco

# confronto tra alcuni metodi, a qualità ridotta
.venv\Scripts\python -m app.cli test --project arco --methods splatfacto inria-3dgs --quality media

# analisi delle foto, escludendo quelle suggerite
.venv\Scripts\python -m app.cli analyze --project arco --photos D:\foto\arco --apply

# solo alcuni metodi e passi
.venv\Scripts\python -m app.cli run --project arco --steps train eval export --methods splatfacto inria-3dgs

# tabella di confronto, anche in CSV
.venv\Scripts\python -m app.cli report --project arco --csv confronto_arco.csv
```

## Dove finiscono i risultati

Nella cartella di lavoro scelta durante l'installazione (voce «Cartella di lavoro» del menu Start);
con l'installazione dal codice sorgente, nella cartella del programma.

```
data\<progetto>\                      allineamento, comune a tutti i metodi
    project.json, photos.json         impostazioni; analisi delle foto ed esclusioni
    alignment.json, split.json        esito dell'allineamento; foto di training e di test
    colmap\                           ricostruzione sparsa; colmap\dense\ per nuvola densa e mesh
    pipeline.log                      log completo delle elaborazioni
outputs\<progetto>\<metodo>\<data>\   un run di training
    run.json, metrics.json            parametri, tempi, versioni; metriche per vista e medie
    test_views\                       alcune viste di test, vere e generate
exports\<progetto>\<metodo>_<data>\   modello esportato
reports\<progetto>\                   report: index.html, grafici, confronto.csv
```

## Sviluppo

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests
```

I test coprono l'analisi delle foto, la suddivisione training/test, la lettura dei log, la
costruzione dei comandi, il registro dei run e i controlli prima dell'avvio; non richiedono la
scheda grafica. `docs\audit.md` descrive le revisioni del programma e i punti ancora aperti.

Il programma di installazione è descritto in `installer\setup.iss` (Inno Setup) e lo costruisce
GitHub Actions (`.github\workflows\release.yml`): avviato a mano produce il Setup come artefatto
dell'esecuzione; con un tag `vX.Y.Z` crea anche la release e vi allega il Setup.

## Software di terze parti

Il repository contiene solo il codice che coordina strumenti esistenti.

| Software | Uso | Licenza |
|---|---|---|
| [COLMAP](https://colmap.github.io/) | allineamento, multi-view stereo, mesh | BSD 3-Clause |
| [nerfstudio](https://docs.nerf.studio/) | training, esportazione, viewer | Apache 2.0 |
| [gsplat](https://docs.gsplat.studio/) | rasterizzazione delle gaussiane | Apache 2.0 |
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria) | implementazione originale di 3D Gaussian Splatting | licenza Inria/MPII, solo ricerca non commerciale |
| [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) | codifiche veloci per NeRF | BSD 3-Clause |
| [PyTorch](https://pytorch.org/) | calcolo su GPU | BSD 3-Clause |
| [PySide6](https://doc.qt.io/qtforpython/) (Qt) | interfaccia grafica | LGPL v3 |
| [FFmpeg](https://ffmpeg.org/) | ridimensionamento delle immagini | GPL v3 (build usata) |
| [SuperSplat](https://github.com/playcanvas/supersplat) | pulizia e pubblicazione dei gaussian splat | MIT |
| [MeshLab](https://www.meshlab.net/) | visualizzazione della mesh | GPL v3 |

### Uso dell'implementazione originale di 3D Gaussian Splatting

Il codice [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
è distribuito da Inria e Max Planck Institut für Informatik con una licenza che ne consente l'uso
**solo per ricerca e valutazione, non commerciale**, e chiede di citare la pubblicazione. L'uso che
se ne fa in questo studio, ricerca accademica non commerciale, rientra in quei termini.

Quel codice non fa parte di questo repository: l'installazione lo scarica dal repository originale,
a un commit fissato. Alla release sono allegati, già compilati, due suoi moduli
(`diff_gaussian_rasterization` e `simple_knn`): restano soggetti alla licenza Inria, allegata
anch'essa, e non alla licenza Apache di questo repository. Chi usa lo strumento per scopi
commerciali deve escludere il metodo `inria-3dgs`.

La pubblicazione va citata in ogni lavoro che usi questi risultati:

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

Il codice di questo repository è distribuito con licenza [Apache 2.0](LICENSE). I componenti di terze
parti mantengono le rispettive licenze, elencate sopra.
