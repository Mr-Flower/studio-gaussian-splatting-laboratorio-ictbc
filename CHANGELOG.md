# Registro delle modifiche

## 0.6.1 — 2026-11-05

- `tab8_geometria` ristretta all'arco: nuove righe «arch only» con le stesse colonne. Il riquadro
  (15,4 × 7,0 × 16,7 m) contiene i rettangoli dei due piloni della verifica della scala e va dal
  suolo a 15,7 m, con 0,5 m di margine. Accuratezza mediana sull'arco: nerfacto 3,7 cm, splatfacto
  4,1, nerfacto-big 6,6, Inria 10,0, splatfacto-big 15,2. Il 90° percentile scende da 46–130 a
  19–53 cm, mentre mediane e ordine dei metodi cambiano poco. Il README è corretto di conseguenza:
  dai dintorni vengono le code, non le differenze tra i metodi.
- `fig5c_esposizione` (in inglese), `fig6c_profilo` e `tab8_geometria` sono nel repository, in
  `paper/foto/`.

## 0.6.0 — 2026-11-04 (terzo piano dell'articolo, PIANO_ARTICOLO_3.md: fasi 13, 12 e 15)

### Cambiamenti che incidono sui risultati

- **Scala verificata in sito** (fase 13, `python -m app.scala --verifica`). La misura di
  riferimento è il basamento di un pilone, 4,33 × 6,013 m, preso con il distanziometro laser sul
  dado. Nella nuvola densa raddrizzata si isolano i due piloni nella fascia tra 1,0 e 2,2 m sopra il
  suolo e si calcola il rettangolo di area minima di ciascuno. La scala verificata è 2,747 m per
  unità (2,741 e 2,753 dai due piloni; 2,668–2,799 dai singoli lati). Le stime dal GPS si scostano
  di +1,5% (orizzontale, 2,789) e −9,4% (tre assi, 2,490). Il rapporto tra i lati nella nuvola
  (1,32–1,33) resta il 4% sotto quello in sito; l'utente l'ha accettato, perché le misure sono
  prese sul dado. Nella fascia 0,2–1,0 m, sulla zoccolatura, lo scarto è del 7–8%. Tutto è in
  `scala.json` sotto `verifica`, con la fonte. `use` = `scale_verified` mette la scala in uso in
  `paper.py` e `geometry.py`. L'altezza misurata è 15,69 m (15,60 m secondo italia.it).
- **Punti di superficie con la stessa procedura per tutti** (fase 12, nuovo `app/surface.py`):
  - 97 viste di training (una su otto) rese a 1320 px con le camere di ciascun metodo;
  - per pixel profondità attesa e opacità accumulata, pixel con opacità > 0,5;
  - un milione di punti con seme 0, senza filtro; a parte la variante con lo stesso filtro
    statistico per tutti;
  - per il codice Inria la profondità si ricava con `override_color`, indipendentemente dalla
    versione.

  `tab8_geometria` riporta superficie, superficie filtrata e centri delle gaussiane (analisi
  secondaria). Nuova figura `fig6c_profilo`, una sezione di 5 cm attraverso il fornice. Accuratezza
  mediana: nerfacto 4,5 cm, splatfacto 5,0, nerfacto-big 8,3, Inria 12,0, splatfacto-big 21,0
  (nella 0.5.0: NeRF 1,9–2,2 cm con `ns-export`, 3DGS 3,1–3,3 cm con i centri). La ρ di RQ4 resta
  +0,80 (p = 0,10), indicata come solo descrittiva.
- Controllo della procedura per i NeRF: i punti di `ns-export pointcloud`, senza filtro e con la
  stessa profondità attesa, distano dalla superficie completa 1,1 cm (nerfacto) e 1,7 cm
  (nerfacto-big) in mediana. Il piano chiedeva meno di 1 cm; l'utente ha accettato lo scarto, che
  viene dalle viste diverse (tutte le foto di training a piena risoluzione contro 97 viste a
  1320 px).
- README (fase 15): abstract, risultati e conclusioni riscritti per la geometria, con la condizione
  «30.000 iterazioni e parametri di densificazione predefiniti» sulla risoluzione piena. L'ora di
  scatto non spiega la correlazione con l'esposizione: ρ(|ΔEV|, ora) = 0,10 e ρ(PSNR corretto,
  ora) da −0,03 a −0,13, non significative; i valori sono in `numeri.json`. Nei limiti la
  procedura unica sostituisce il confronto non alla pari; la sezione Scala descrive la verifica.

### Novità

- `fig5c_esposizione` con le etichette in inglese.
- Sito: didascalie della geometria aggiornate e figura `fig6c_profilo`.
- Test senza scheda grafica: ritroiezione su una camera sintetica nelle due convenzioni di
  profondità, filtro statistico con outlier noti, rettangolo di area minima e separazione dei
  piloni.

## 0.5.0 — 2026-10-23 (secondo piano dell'articolo, PIANO_ARTICOLO_2.md)

### Cambiamenti che incidono sui risultati

- **Confronto geometrico con la fotogrammetria** (fase 10, `app/geometry.py`): accuratezza e
  completezza dei centri delle gaussiane opache e delle nuvole dei NeRF rispetto alla nuvola densa
  di COLMAP, in metri; tabella `tab8_geometria`, figure `fig6_distanze` e `fig6b_distanze_pianta`,
  correlazione tra qualità d'immagine e accuratezza. Opzione `--mvs-max-side` per il lato delle
  immagini della nuvola densa (sul caso di studio 1600 px, per lo spazio su disco).
- README: risultati e conclusioni riscritti con valori grezzi e corretti, effetto della risoluzione
  di training, esposizione, disegno 2×2 di nerfacto e geometria; nuovi limiti (riferimento
  geometrico, assenza di maschera, vettore di aspetto sulle viste di test).

- **Metriche con correzione del colore** (fase 7): per ogni vista anche PSNR, SSIM e LPIPS dopo
  una correzione affine per canale (`psnr_cc`, `ssim_cc`, `lpips_cc`, con i coefficienti), che
  assorbe le differenze globali di esposizione e di bilanciamento del bianco. Usa l'immagine vera:
  è un limite superiore, da riportare accanto ai valori grezzi.
- **Scale di valutazione comuni** (fase 8): con `--eval-side N` le metriche si calcolano anche
  riducendo entrambe le immagini a lato N (`at_side` in `metrics.json`), per separare risoluzione
  di training e risoluzione di misura.

### Novità

- Comando `rivaluta`: rivaluta i run già fatti senza rifare il training (salta quelli che hanno
  già tutte le misure chieste).
- `app/paper.py`: colonne corrette in tabella 3 e 4 (Holm separato per grezze e corrette),
  distanza di esposizione delle viste dalle EXIF (EV100), figura `fig5c_esposizione`, tabella
  `tab6_risoluzione` (risoluzione di training per scala di valutazione) e `tab7_opzioni_nerfstudio`
  (disegno 2×2 di correzione delle pose e vettore di aspetto).
- Script dei run: passi 10–14 (rivalutazione, due run di nerfacto per il 2×2, report e materiali).
- Pulizia dei modelli più severa: toglie anche le gaussiane color cielo e quelle staccate dalla
  superficie; i modelli da pubblicare hanno l'alto verso −y, come si aspetta SuperSplat.
- Test che impedisce caratteri di controllo in README, CHANGELOG e piani; corretto di nuovo un
  carattere di controllo nel README.

## 0.4.1 — 2026-10-22

- **Pulizia automatica dei modelli** (`python -m app.clean`): toglie gaussiane quasi trasparenti,
  enormi, isolate o fuori dalla zona del volo e scrive `splat_pulito.ply` accanto all'esportato,
  con i conteggi (`pulizia.json`) e un'anteprima in pianta, prospetto e fianco. L'originale non
  cambia e le misure restano quelle del modello originale.
- **Sito dei risultati** (`python -m app.site`): pagina di presentazione, report completo e schede
  dei modelli. I file dei modelli non vanno nel sito, che li renderebbe scaricabili: si preparano a
  parte (`exports/<progetto>/da_pubblicare/*.splat`, puliti e nel sistema di riferimento di COLMAP)
  per un visualizzatore esterno, il cui indirizzo si passa con `--viewer metodo=indirizzo`.
  Pubblicato per il caso di studio su GitHub Pages.

## 0.4.0 — 2026-10-22

### Cambiamenti che incidono sui risultati

- **Correzione delle pose spenta per default** nei metodi nerfstudio: nerfacto e nerfacto-big la
  avevano attiva, ma la valutazione usa le pose di COLMAP, quindi i NeRF erano penalizzati. Opzione
  «Correggi le pose» in «Opzioni avanzate», `--pose-refinement` da riga di comando.
- **Vettore di aspetto per foto spento per default** nei NeRF: compensava differenze di luce e di
  sviluppo che il 3DGS non può compensare. Opzione in «Opzioni avanzate», `--appearance-embedding`.
- Le due opzioni sono passate sempre in modo esplicito a `ns-train`, registrate in `run.json` e
  fanno parte delle condizioni con cui il report decide quali run confrontare. I run precedenti
  valgono con i default di allora: NeRF con entrambe attive, 3DGS con entrambe spente.

### Novità

- **Verifica delle immagini di riferimento** (`python -m app.check_gt --project <nome>`): riproietta
  l'immagine vera del codice Inria nella camera di nerfstudio e misura PSNR e SSIM sulla zona
  comune, vista per vista (`reports/<progetto>/check_gt.json`). Sul caso di studio, a risoluzione
  piena: PSNR medio 42,6 dB (minimo 38,0), SSIM 0,984 su 111 viste.
- **Materiali per l'articolo** (`python -m app.paper --project <nome> --out paper\<nome>`): tabelle
  in CSV e LaTeX (dati del rilievo, configurazioni, risultati con intervalli bootstrap, confronti
  appaiati vista per vista con test di Wilcoxon e correzione di Holm, fotogrammetria), figure in PDF
  e PNG (camere viste dall'alto, qualità rispetto all'ora di scatto, JPG contro DNG) e `numeri.json`
  con tutti i numeri citabili. Legge solo i file già prodotti. Con più run dello stesso metodo nelle
  stesse condizioni usa la media per vista tra i run.
- **Ripetizioni con seme** (`--seed N` in `run` e `test`): nerfstudio lo riceve con `--machine.seed`, il
  codice Inria tramite `app/inria_avvio.py`, che reimposta il seme subito dopo `safe_state` senza
  modificare il codice. Il seme usato è in `run.json` (senza `--seed`: 42 per nerfstudio, 0 per
  Inria). Report e `app/paper.py` raggruppano i run che differiscono solo per il seme. Lo stesso
  seme non dà risultati identici: sulla scheda grafica restano differenze (su 40 foto, splatfacto
  18,40 e 18,03 dB con lo stesso seme; Inria 19,33 e 19,33).
- **Scala metrica dal GPS del drone** (`python -m app.scala --project <nome>`): allinea una copia del
  modello sparso alle posizioni GPS delle EXIF, senza toccare l'allineamento dei run, e salva in
  `data/<progetto>/scala.json` due stime: quella di `model_aligner` sui tre assi e una dalle sole
  posizioni orizzontali, perché la quota del GPS è poco affidabile. Sul caso di studio: 2,49 e
  2,79 m per unità (IC 95% 2,76–2,82); residui orizzontali di circa 1 m, verticali di 11 m (d.s.).
  `app/paper.py` usa la stima indicata nel campo `use`, se c'è.
- Script `scripts/run_articolo.ps1`: i run definitivi per l'articolo in sequenza, con registro.
- Il controllo delle immagini di riferimento usa i run del confronto principale e, con
  `--downscale`, quelli a una risoluzione ridotta (`check_gt_<fattore>.json`).
- README: risultati del caso di studio a risoluzione piena, effetto delle opzioni di nerfstudio e
  della risoluzione, ripetibilità.
- Lettura di camere e pose dai modelli COLMAP (`colmap_model.read_cameras`, `read_poses`).
- Picco di memoria della scheda grafica durante ogni training (`gpu_peak_mb` in `run.json`, letto
  con `nvidia-smi` ogni 2 secondi; comprende gli altri programmi che usano la scheda).

### Correzioni

- README: nel comando di conversione un carattere di controllo aveva sostituito `\a`.

## 0.3.1 — 2026-10-02

- **Il modello si può guardare in qualsiasi momento.** Il viewer dei modelli già creati si apre
  anche mentre è in corso un'altra elaborazione. Durante il training Inria il modello viene
  salvato dieci volte (una ogni decimo delle iterazioni; resta solo l'ultimo salvataggio) e
  «Guarda il training in corso» lo apre in SuperSplat.
- **SuperSplat si apre con il modello già caricato**: non serve più trascinare il file. Il
  programma lo consegna al browser da un indirizzo locale (127.0.0.1), raggiungibile solo dal
  computer stesso; il browser può chiedere il permesso di accedere alla rete locale.
- Un training Inria interrotto lascia l'ultimo salvataggio intermedio, che compare tra i run con
  il numero di iterazioni raggiunto.

## 0.3.0 — 2026-10-02

### Novità

- **Interfaccia più semplice**: il metodo si sceglie da un menu a tendina e «Crea il modello 3D»
  fa tutto il resto (allineamento se manca, training, valutazione, esportazione). I passi non si
  spuntano più a mano; le opzioni meno usate sono in «Opzioni avanzate».
- **Qualità**: quattro livelli (massima, alta, media, bozza). Si parte dalla massima, cioè dalle
  foto a risoluzione piena; i livelli inferiori servono solo a fare prima. Da riga di comando:
  `--quality` e `--max-side`.
- **Foto lette dal disco**: quando le foto non entrano in memoria tutte insieme, il training le
  legge dal disco invece di ridurne la risoluzione (gsplat, codice Inria e NeRF). Più lento, ma
  la risoluzione non dipende più dalla memoria del computer.
- **Test con scelta dei metodi**: «Test: confronta più metodi…» apre l'elenco dei metodi da
  spuntare e la qualità comune a tutti. Da riga di comando: `test --methods`.
- **Modello di camera automatico**: una camera per sottocartella se le foto sono divise in
  sottocartelle, altrimenti una sola.
- **Programma di installazione**: la release contiene `GaussianSplatting-Setup-<versione>.exe`, una
  procedura guidata che copia il programma in `Programmi\Gaussian Splatting`, chiede la cartella
  di lavoro e l'icona sul desktop, scarica e configura i componenti mostrando l'avanzamento, e
  registra la disinstallazione. Lo costruisce GitHub Actions e lo allega alla release.
- **Cartella di lavoro separata**: con il programma installato, progetti e risultati stanno in una
  cartella scelta dall'utente e non in quella del programma.
- **Altri formati**: dal modello selezionato si ottengono la nuvola di punti (`.ply`) e un `.glb`;
  la mesh della fotogrammetria si converte in `.glb` (pulsante «Altri formati…» e `python -m app.convert`).
- Icona del programma, anche nella barra delle applicazioni.

### Cambiamenti che incidono sui risultati

- **La risoluzione predefinita non è più limitata a 1600 pixel**: è quella piena. Per riavere la
  regola precedente: qualità «Media».
- **Con le foto lette dal disco il tempo di training include la lettura**, e i NeRF campionano i
  raggi da un gruppo di foto alla volta. I run registrano il tipo di caricamento (`photos_in`).
- La valutazione del metodo Inria legge una foto alla volta invece di caricarle tutte: i valori
  non cambiano, serve meno memoria.
- **Il report confronta solo run fatti nelle stesse condizioni** (risoluzione e iterazioni): prima
  prendeva l'ultimo run di ogni metodo, anche se a risoluzioni diverse.

### Correzioni

- Se la memoria si esaurisce durante un passo, il messaggio lo dice e invita ad abbassare la qualità.
- Metodo Inria: oltre i 1600 pixel di larghezza il codice Inria riduceva le foto da solo, quindi con
  «metà risoluzione» o «risoluzione piena» avrebbe lavorato a una risoluzione diversa dagli altri
  metodi. Ora la risoluzione gli è imposta. I run fatti con la 0.2.0 alla risoluzione automatica
  (fino a 1600 pixel) non erano interessati.
- I moduli compilati sono dentro il Setup; `installa.ps1` li scarica, se mancano, dalla release
  che li contiene e non da quella con il numero di versione del programma.
- Il limite sulla lunghezza del percorso di installazione passa da 60 a 75 caratteri (misurato sui
  percorsi più lunghi dei pacchetti installati).

## 0.2.0 — 2026-10-02

### Novità

- **Analisi delle foto**: nitidezza, esposizione e quasi-doppioni, con proposta delle foto da
  escludere dall'allineamento (scheda «Foto» e comando `analyze`).
- **Gaussian splatting originale di Inria** come metodo in più, accanto a gsplat.
- **Modalità test**: un solo comando allena, valuta ed esporta tutti i metodi installati e genera
  il report (pulsante «Test: confronta tutti i metodi» e comando `test`).
- **Report del confronto** in HTML: grafici di qualità, costo e qualità rispetto al tempo, tabella
  delle misure, differenze di processo, viste di test affiancate.
- **Installazione automatica**: `Installa.bat` scarica e configura Python, i pacchetti, i moduli
  CUDA già compilati, COLMAP, FFmpeg, MeshLab e il codice Inria.
- Licenza Apache 2.0.

### Cambiamenti che incidono sui risultati

- **Suddivisione training/test comune a tutti i metodi**: una foto ogni otto, in ordine di nome,
  al posto del 10% scelto da nerfstudio. I run fatti con la versione 0.1.0 non sono confrontabili
  con i nuovi.
- **Valutazione unica**: PSNR, SSIM, LPIPS e tempo di rendering sono calcolati da un solo modulo
  (`app/evaluate.py`) per tutti i metodi, non più dal programma che ha allenato il modello.
- **Risoluzione sempre esplicita** e uguale per tutti i metodi.
- Ambiente aggiornato a PyTorch 2.4.1 con CUDA 12.4; tiny-cuda-nn installato, quindi i metodi NeRF
  non usano più l'implementazione lenta.

### Correzioni

- Il programma usa gli strumenti dell'ambiente Python con cui è avviato, non un percorso fisso.
- L'avanzamento del training si legge anche quando un'iterazione dura più di un secondo.
- L'indirizzo del viewer si ricava anche dall'output di `ns-viewer`, che lo scrive in un formato diverso.

## 0.1.0 — 2026-10-02

Prima versione pubblicata. Riscrittura del prototipo iniziale a seguito della revisione descritta
in `docs/audit.md`.

### Novità

- Confronto di più metodi sullo stesso allineamento: gaussian splatting (`splatfacto`,
  `splatfacto-big`) e NeRF (`nerfacto`, `nerfacto-big`), oltre alla fotogrammetria classica con COLMAP.
- Valutazione di ogni run sulle viste escluse dal training: PSNR, SSIM, LPIPS.
- Tabella di confronto dei run, nell'interfaccia e da riga di comando, esportabile in CSV.
- Ogni run registra parametri, tempi, scheda grafica e versioni dei componenti (`run.json`).
- Riga di comando (`python -m app.cli`) con la stessa pipeline dell'interfaccia grafica.
- Riepilogo dell'allineamento: foto allineate, punti, errore di riproiezione.

### Correzioni

- Rifare l'allineamento non cancella più quello esistente finché il nuovo non è completo.
- I run allenati su un allineamento superato sono segnalati ed esclusi da viewer e valutazione.
- L'indirizzo del viewer del training è letto dal log, invece di assumere la porta 7007.
- Un progetto non può essere elaborato da due finestre contemporaneamente.
- Gli errori imprevisti dell'interfaccia sono salvati in `logs/app-errors.log`.

### Rimosso

- Script PowerShell `1_sfm`, `2_train`, `3_export`, `4_mesh` e `activate.ps1`, sostituiti dalla riga di comando.
