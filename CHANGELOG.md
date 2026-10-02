# Registro delle modifiche

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
