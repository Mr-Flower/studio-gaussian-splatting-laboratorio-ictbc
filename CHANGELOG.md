# Registro delle modifiche

## 0.3.0 — 2026-10-02

### Novità

- **Interfaccia più semplice**: il metodo si sceglie da un menu a tendina e «Crea il modello 3D»
  fa tutto il resto (allineamento se manca, training, valutazione, esportazione). I passi non si
  spuntano più a mano; le opzioni meno usate sono in «Opzioni avanzate».
- **Qualità**: quattro livelli (massima, alta, media, bozza). Si parte dalla massima, cioè dalla
  risoluzione più alta con cui le foto entrano nella memoria del computer; i livelli inferiori
  servono solo a fare prima. Da riga di comando: `--quality` e `--max-side`.
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
- Icona del programma, anche nella barra delle applicazioni.

### Cambiamenti che incidono sui risultati

- **La risoluzione predefinita non è più limitata a 1600 pixel**: è la più alta che entra in
  memoria per tutti i metodi allenati insieme. Per riavere la regola precedente: qualità «Alta».
- **Il report confronta solo run fatti nelle stesse condizioni** (risoluzione e iterazioni): prima
  prendeva l'ultimo run di ogni metodo, anche se a risoluzioni diverse.

### Correzioni

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
