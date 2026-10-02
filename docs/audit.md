# Audit del programma (prima della versione 0.1.0)

Revisione del prototipo iniziale (un'interfaccia che lanciava allineamento, training di un solo
metodo, esportazione e mesh), fatta rispetto allo scopo del repository: confrontare metodi di
ricostruzione 3D da fotografie in modo ripetibile. Per ogni problema è indicato come è stato risolto
nella versione 0.1.0.

Gravità: **A** = può far perdere lavoro o produrre risultati sbagliati, **M** = limita l'uso per la
ricerca o l'affidabilità, **B** = qualità del codice e dell'interfaccia.

## Adeguatezza allo scopo

| # | Gravità | Problema | Correzione |
|---|---|---|---|
| 1 | A | Nessuna misura di qualità: senza metriche il confronto tra metodi è solo visivo. | Nuovo passo di valutazione (`ns-eval`) con PSNR, SSIM e LPIPS sulle viste escluse dal training. |
| 2 | A | Un solo metodo per volta e solo gaussian splatting: NeRF non era previsto. | Selezione di più metodi (gaussian splatting e NeRF), allenati in sequenza sullo stesso allineamento. |
| 3 | A | Ogni esportazione sovrascriveva `exports/<progetto>/splat.ply`: impossibile tenere due risultati. | Ogni run ha la propria cartella di esportazione, `exports/<progetto>/<metodo>_<data>/`. |
| 4 | M | Tempi di calcolo non registrati. | Durata di ogni passo salvata in `run.json` (training, valutazione, esportazione) e `mesh.json`. |
| 5 | M | Nessuna traccia di parametri e versioni: un risultato non era riproducibile. | `run.json` registra metodo, iterazioni, risoluzione, numero di foto, scheda grafica e versioni di nerfstudio, gsplat, PyTorch e COLMAP. |
| 6 | M | Nessuna vista d'insieme dei risultati. | Scheda «Confronto dei metodi» con una riga per run ed esportazione in CSV; stesso contenuto da riga di comando (`report`). |
| 7 | M | La pipeline esisteva due volte (script PowerShell e interfaccia), con differenze: gli script usavano sempre il modello `sparse/0`, l'interfaccia quello con più foto. | Un solo nucleo, usato sia dall'interfaccia sia dalla nuova riga di comando; script PowerShell rimossi. |

## Correttezza e robustezza

| # | Gravità | Problema | Correzione |
|---|---|---|---|
| 8 | A | Rifare l'allineamento cancellava subito quello esistente: un errore o un'interruzione lasciavano il progetto senza allineamento. | Il nuovo allineamento è calcolato in una cartella di lavoro e sostituisce il precedente solo quando è completo. |
| 9 | A | Dopo un nuovo allineamento, i training precedenti restavano utilizzabili ma non erano più coerenti con le pose delle camere (viewer e valutazione avrebbero dato risultati sbagliati). | Ogni run registra l'impronta dell'allineamento usato; i run con allineamento superato sono segnalati in tabella ed esclusi da viewer, valutazione ed esportazione. |
| 9b | A | nerfstudio salva le varianti «-big» nella cartella del metodo base (`splatfacto-big` in `splatfacto/`): i run dei due metodi si sarebbero confusi nella tabella. | Il nome del metodo è imposto all'avvio del training, così ogni metodo ha la propria cartella. |
| 10 | A | La conferma per rifare l'allineamento non spiegava le conseguenze e aveva «Sì» come scelta predefinita. | Messaggio esplicito su tempi e conseguenze, con «No» predefinito. |
| 11 | M | Il pulsante «Guarda il training» apriva sempre la porta 7007; se occupata, nerfstudio ne sceglie un'altra e si apriva la pagina sbagliata. | L'indirizzo del viewer è letto dal log del training. |
| 12 | M | «Visualizza il modello» restava attivo durante un'elaborazione: avviava un secondo processo sulla GPU e occupava la porta del viewer. | Pulsante disattivato durante l'elaborazione; il viewer aperto viene chiuso all'avvio. |
| 13 | M | Niente impediva due elaborazioni contemporanee sullo stesso progetto. | File di blocco per progetto, con riconoscimento dei blocchi lasciati da processi terminati. |
| 14 | M | Lo stato della scheda grafica veniva letto con una chiamata bloccante nel thread dell'interfaccia. | Lettura asincrona. |
| 15 | M | Avviata senza console, l'applicazione non mostrava gli errori imprevisti. | Gli errori sono salvati in `logs/app-errors.log` e mostrati in una finestra; all'avvio si verifica la presenza dei componenti esterni. |
| 16 | M | Lo script di training offriva il metodo `splatfacto-mcmc`, che nerfstudio 1.1.5 non ha. | Elenco dei metodi limitato a quelli verificati su questa installazione. |
| 17 | M | Mesh ad alta risoluzione su centinaia di foto avviabile senza avviso, con tempi di oltre un giorno. | Richiesta di conferma con il numero di foto e la risoluzione scelta. |
| 18 | B | Il log a video era occupato da decine di migliaia di righe della tabella di training. | Quelle righe aggiornano solo lo stato; il log completo resta su file. |
| 19 | B | L'avanzamento del training non si leggeva per la prima riga dei metodi NeRF e quando un'iterazione dura più di un secondo (formato diverso). | Parser corretto e verificato su righe reali di tutti i metodi. |
| 20 | B | Le spunte dei passi venivano reimpostate a ogni carattere digitato nel nome del progetto, e il conteggio delle foto scorreva la cartella a ogni tasto. | Spunte reimpostate solo al cambio di progetto; conteggio ritardato. |
| 21 | B | Nessun riepilogo dell'allineamento: non si sapeva quante foto fossero state allineate. | Passo di verifica con foto allineate, punti ed errore di riproiezione, mostrati nell'interfaccia. |

## Qualità del codice

| # | Gravità | Problema | Correzione |
|---|---|---|---|
| 22 | M | Nessun test automatico. | 35 test sui parser dei log, sulla costruzione dei comandi, sul registro dei run e sui controlli; eseguiti a ogni push con GitHub Actions. |
| 23 | B | Un unico file per finestra, esecuzione e logica. | Moduli separati: `config`, `progress`, `runs`, `pipeline`, `runner`, `window`, `cli`. |
| 24 | B | Dipendenze non fissate (`rawpy`, `PySide6`), nessun numero di versione. | Versioni fissate in `requirements.txt`; versione dell'applicazione in `app/__init__.py`. |

## Punti aperti

- **Licenza del repository**: non è stata scelta. Senza licenza altri non possono riusare il codice.
- **Metriche geometriche**: il confronto misura la qualità delle immagini sintetizzate, non
  l'accuratezza della geometria. La mesh fotogrammetrica non ha quindi metriche confrontabili con
  gli altri metodi; servirebbe un riferimento metrico (ad esempio un rilievo laser).
- **NeRF senza tiny-cuda-nn**: funziona con l'implementazione PyTorch, più lenta. Installare
  tiny-cuda-nn richiede un compilatore C++ e i diritti di amministratore.
- **Variazioni di esposizione**: nessuno dei metodi configurati compensa differenze di luce tra le
  foto; su rilievi lunghi questo penalizza le metriche.
- **Installer**: l'applicazione si usa dalla cartella del repository; non esiste ancora un programma
  di installazione.
