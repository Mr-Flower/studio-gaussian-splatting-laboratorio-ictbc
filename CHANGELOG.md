# Registro delle modifiche

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
