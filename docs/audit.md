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
| 22 | M | Nessun test automatico. | Test automatici (51 alla versione 0.2.0) sui parser dei log, sulla costruzione dei comandi, sul registro dei run e sui controlli; eseguiti a ogni push con GitHub Actions. |
| 23 | B | Un unico file per finestra, esecuzione e logica. | Moduli separati: `config`, `progress`, `runs`, `pipeline`, `runner`, `window`, `cli`. |
| 24 | B | Dipendenze non fissate (`rawpy`, `PySide6`), nessun numero di versione. | Versioni fissate in `requirements.txt`; versione dell'applicazione in `app/__init__.py`. |

## Seconda revisione (versione 0.2.0)

Riguarda la validità del confronto, dopo l'aggiunta del gaussian splatting originale di Inria.

| # | Gravità | Problema | Correzione |
|---|---|---|---|
| 26 | A | nerfstudio e il codice Inria sceglievano foto di test diverse (il 10% a intervalli regolari il primo, una su otto il secondo): le metriche dei due non erano confrontabili. | Un'unica suddivisione per progetto, scritta nei formati letti da entrambi; in valutazione si verifica che le viste siano proprio quelle. |
| 27 | A | Le metriche erano calcolate dal programma di training: LPIPS con reti diverse (AlexNet in nerfstudio, VGG nel codice Inria) e tempi di rendering misurati in modi diversi. | Un solo modulo di valutazione per tutti i metodi. |
| 28 | A | La risoluzione «automatica» era decisa da ciascun programma con regole diverse (lato massimo 1600 px per riduzioni successive in nerfstudio, ridimensionamento a 1600 px nel codice Inria). | Fattore di riduzione calcolato una volta e passato in modo esplicito a tutti. |
| 29 | M | Tutte le foto della cartella entravano nell'allineamento, comprese quelle mosse, quasi duplicate o estranee al soggetto. | Analisi preliminare con proposta di esclusione. |
| 30 | M | Una prima misura di nitidezza (varianza del laplaciano) segnalava come mosse le foto con molto cielo o superfici lisce: 140 foto su 906 nel caso di studio. | Misura indipendente dal soggetto (dettaglio fine rispetto al grossolano, sulle sole zone con dettaglio), confrontata tra foto della stessa cartella: 9 segnalate, verificate a vista. |
| 31 | M | Il programma cercava gli strumenti in un percorso fisso (`.venv`). | Usa l'ambiente con cui è avviato. |
| 32 | M | L'installazione richiedeva di scaricare a mano cinque componenti e, per tiny-cuda-nn e il codice Inria, un compilatore C++ e CUDA Toolkit. | Installazione automatica con moduli già compilati. |
| 33 | B | Il pulsante del viewer non riconosceva l'indirizzo scritto da `ns-viewer`. | Riconosciuti entrambi i formati. |

## Punti aperti

- **Metriche geometriche**: il confronto misura la qualità delle immagini sintetizzate, non
  l'accuratezza della geometria. La mesh fotogrammetrica non ha quindi metriche confrontabili con
  gli altri metodi; servirebbe un riferimento metrico (ad esempio un rilievo laser).
- **Immagini di riferimento**: nerfstudio e il codice Inria correggono la distorsione con procedure
  diverse; le immagini vere usate nella valutazione coincidono nel contenuto ma non pixel per pixel.
- **Variazioni di esposizione**: nessuno dei metodi configurati compensa differenze di luce tra le
  foto; su rilievi lunghi questo penalizza le metriche.
- **Foto estranee al soggetto**: l'analisi non le riconosce (nel caso di studio, una foto di gruppo
  è stata tolta a mano).
- **Ripetizioni**: ogni configurazione è eseguita una volta; la variabilità tra esecuzioni non è stimata.
- **Installazione**: provata su una sola macchina. Lo scaricamento dei moduli compilati dalla
  release richiede che il repository sia pubblico.
- **Palette dei grafici**: è quella di riferimento, documentata come distinguibile anche con
  daltonismo; il controllo automatico non è stato eseguito su questa macchina.
