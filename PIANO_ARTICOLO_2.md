# Piano di lavoro per l'articolo — seconda parte (dopo i run della versione 0.4.0)

> Va nella radice del repository, accanto a `PIANO_ARTICOLO.md`, di cui continua la numerazione
> delle fasi. Le regole sono le stesse: test verdi senza GPU, una voce nel `CHANGELOG.md` per ogni
> fase, nessuna modifica al codice Inria o ai pacchetti installati, e una domanda all'utente per
> ogni scelta di metodo non scritta qui.
>
> **Le fasi 7–9 non richiedono nuovi training**, salvo due run brevi di nerfacto: si rivalutano i
> modelli già allenati. Le **conclusioni del README non vanno toccate** finché le fasi 7 e 8 non
> sono finite, perché possono cambiare la classifica tra i metodi.

---

## Perché

Dall'esame dei risultati (versione 0.4.0, progetto `foto`) emergono tre problemi.

**1. L'esposizione domina le metriche.** Le foto sono state scattate tra le 9:25 e le 16:08 con
esposizione e bilanciamento del bianco automatici. Un modello senza compensazione rende un aspetto
"medio", e sulle viste di test fotografate più chiare o più scure il PSNR crolla per una ragione
che non riguarda il metodo. Una stima sulle miniature delle tre viste mostrate nel report (solo
indicativa: immagini da circa 210 px, JPEG):

| Vista | Luminosità media vera / generata | PSNR grezzo | PSNR dopo una correzione affine del colore per canale |
|---|---|---|---|
| 1 | 0,78 / 0,57–0,62 | 12,3–14,8 dB | 21,0–26,1 dB |
| 2 | 0,47 / 0,43–0,47 | 21,7–23,2 dB | 22,5–23,4 dB |
| 3 | 0,51 / 0,46–0,50 | 21,2–23,9 dB | 22,6–24,3 dB |

Sulla vista 1 la correzione cambia anche la classifica: il codice Inria passa da quasi primo a
ultimo. Le medie su 111 viste mescolano quindi due effetti, la fedeltà del metodo e la distanza di
esposizione della vista dalla media. Vanno separati.

**2. La tabella sulla risoluzione confonde due cose.** Passando da 5280 a 1320 px cambiano insieme
la risoluzione di training e quella a cui si misura. Così non si può dire se allenare a piena
risoluzione serva. Bisogna valutare tutti i modelli anche a una scala di valutazione comune.

**3. Il run di nerfacto con le opzioni di nerfstudio confonde due opzioni.** Il calo a 15,11 dB
viene dalla correzione delle pose, che peggiora le viste di test, e dal vettore di aspetto
attivati insieme. Il vettore di aspetto, da solo, potrebbe invece migliorare le metriche proprio
per il punto 1. Serve un disegno 2×2.

---

## Fase 7 — Metriche con correzione del colore e dati di esposizione (solo rivalutazione)

1. **`app/evaluate.py`.** Per ogni vista, oltre a PSNR, SSIM e LPIPS grezzi, si calcolano le
   stesse metriche dopo una **correzione affine per canale**. Per ciascun canale c si trovano a e
   b minimizzando ‖a·pred_c + b − gt_c‖² su tutti i pixel (minimi quadrati in forma chiusa, in
   float64 sulla GPU), poi si applica `clamp(0, 1)`. Si salvano `psnr_cc`, `ssim_cc`, `lpips_cc` e
   i sei coefficienti per vista in `metrics.json` (`per_image`), e le medie con dev. st. in
   `results`. I nomi esistenti restano invariati, così i run vecchi restano leggibili.
   - Nel docstring si scrive che la correzione usa l'immagine vera ed è quindi un limite
     superiore: assorbe le differenze globali di esposizione e di bilanciamento del bianco, non
     gli errori locali di colore. Nell'articolo vanno riportati sia i valori grezzi sia quelli
     corretti.
   - Si aggiunge un test senza GPU (tensori CPU) della funzione di correzione: un'immagine
     trasformata con un a e un b noti torna identica.
2. **Esposizione delle foto.** In `app/paper.py` si leggono dalle EXIF delle foto di test
   ExposureTime (0x829A), FNumber (0x829D) e ISOSpeedRatings (0x8827), dall'IFD Exif. Per i DNG
   sviluppati, se mancano, si prendono dal DNG originale o si lascia vuoto. Si calcola
   EV100 = log2(N²/t) − log2(ISO/100), e per ogni vista la **distanza di esposizione**
   ΔEV = EV100 − mediana di EV100 delle foto di training.
3. **Comando di rivalutazione** senza training, per tutti i run confrontabili:
   `cli run --project foto --steps eval --methods ... --quality <q>`. Se il passo eval rifiuta i
   run già valutati, si aggiunge `--force-eval`. Va lanciato su tutti i run delle fasi precedenti
   (principali, risoluzioni, semi, nerfacto con le opzioni).
4. **`app/paper.py`**:
   - `tab3_risultati` riporta, per ogni metrica, sia la colonna grezza sia quella corretta;
     `tab4_appaiati` si calcola su entrambe;
   - nuova figura `fig5c_esposizione`: PSNR grezzo e corretto per vista rispetto a |ΔEV|, un
     pannello per metodo, con la retta di regressione e la ρ di Spearman nella legenda;
   - in `numeri.json` vanno ρ e p per metodo, la differenza media grezzo−corretto e il numero di
     viste in cui la classifica tra 3DGS Inria e splatfacto si inverte con la correzione.
5. **Maschera opzionale del monumento** (solo se costa poco). Le metriche si ricalcolano escludendo
   il cielo, cioè i pixel in cui l'opacità accumulata del modello 3DGS è sotto 0,5, oppure un
   semplice criterio di colore validato a occhio su 10 viste. Se non si trova una soluzione
   robusta in mezza giornata, si lascia stare e lo si scrive tra i limiti.

## Fase 8 — Scala di valutazione comune (solo rivalutazione)

1. **`app/evaluate.py`**, opzione `--eval-side N` (ripetibile). Dopo `finish()`, l'immagine
   generata e quella vera si riducono **entrambe** a lato maggiore N con
   `torch.nn.functional.interpolate(..., mode="area")` e si ricalcolano tutte le metriche, grezze
   e corrette. Si salvano in `metrics.json` sotto `at_side: {"1320": {...}, "2640": {...}}`.
2. Si rivalutano con `--eval-side 2640 --eval-side 1320` tutti i run alle tre risoluzioni di
   training.
3. In `app/paper.py`, nuova tabella `tab6_risoluzione`: righe = risoluzione di training
   (5280/2640/1320), colonne = scala di valutazione (5280/2640/1320). Celle con PSNR, SSIM e
   LPIPS corretti, per i due 3DGS, più il numero di gaussiane, il training in ore e il picco di
   memoria per riga. La diagonale riproduce la tabella attuale del README.
   - La domanda a cui risponde, da scrivere in `numeri.json`, è questa: a parità di scala di
     valutazione (1320 px), il modello allenato a 5280 px è migliore di quello allenato a 1320 px?
     Si riportano la differenza e il Wilcoxon appaiato.
   - Si annota anche il numero di gaussiane per risoluzione di training. Circa 0,7 milioni a 20
     megapixel sono pochi: se il numero cresce poco con la risoluzione, il modello ha poca
     capacità a piena risoluzione e questo spiega l'LPIPS alto. Va solo misurato e riportato,
     senza cambiare i parametri di densificazione.

## Fase 9 — Disegno 2×2 per nerfacto (due training brevi, circa 0,6 h ciascuno)

1. Due run in più, a qualità massima:
   - `--appearance-embedding` da solo (pose spente);
   - `--pose-refinement` da solo (aspetto spento).

   Con il principale (entrambe spente) e quello già fatto (entrambe accese) il 2×2 è completo.
2. Con il vettore di aspetto acceso, le viste di test usano il vettore medio, come fa nerfstudio
   per default. Va scritto tra i limiti che la procedura di NeRF-W (ottimizzare il vettore su metà
   dell'immagine di test e misurare sull'altra metà) non è stata applicata.
3. In `app/paper.py`, nuova tabella `tab7_opzioni_nerfstudio`: 2×2 con metriche grezze e
   corrette, e l'effetto principale di ciascuna opzione.
4. In `scripts/run_articolo.ps1` si aggiungono i due run come passi numerati, senza rinumerare
   quelli esistenti: vanno in fondo, prima dei materiali.

## Fase 10 — Fotogrammetria e confronto geometrico (solo se l'utente conferma)

Titolo e abstract parlano di tre famiglie di metodi, ma la fotogrammetria non è stata eseguita.
Le alternative sono due, e **sceglie l'utente**: o la si esegue qui, oppure titolo e abstract si
restringono a 3DGS e NeRF. In questo secondo caso la fase si salta e se ne informa l'utente.

1. **Nuvola densa a risoluzione ridotta**, per tenere i tempi sotto controllo:
   `patch_match_stereo` con `--PatchMatchStereo.max_image_size 2640` e `--PatchMatchStereo.geom_consistency true`,
   poi `stereo_fusion` e `poisson_mesher`. Si aggiunge l'opzione `--mvs-max-side` in `cli`, e il
   valore va registrato in `mesh.json`. Prima di lanciare si stima il tempo su 40 foto e lo si
   comunica.
2. **Confronto geometrico** (`app/geometry.py`, nuovo), con la nuvola densa come **riferimento,
   non come verità**:
   - nuvole confrontate: i centri delle gaussiane con opacità > 0,5 per ciascun 3DGS, e il
     `point_cloud.ply` di nerfacto e nerfacto-big;
   - area: un riquadro attorno ai punti della nuvola sparsa, ridotto ai percentili 2–98 su ogni
     asse, così si escludono cielo e intorno lontano;
   - misure: distanza da ogni punto al punto più vicino della nuvola densa (`scipy.spatial.cKDTree`,
     su un sottocampione di 2 milioni di punti), mediana, 90° percentile e frazione entro 5/10/20
     cm. Il tutto convertito in metri con la scala di `scala.json`, e nei due versi (completezza e
     accuratezza);
   - figure: istogrammi delle distanze per metodo e una vista dall'alto colorata per distanza.
3. **Scala.** Le due stime di `app/scala.py` (2,49 e 2,79 m per unità) differiscono dell'11%.
   Prima di usare i metri nell'articolo si chiede all'utente **una misura nota del monumento**,
   per esempio la larghezza del fornice presa da un rilievo o con una misura diretta, e la si
   confronta con la stessa distanza misurata sul modello. Il risultato va in `scala.json`
   (`verifica`).
4. Con questa fase si può rispondere a RQ4: la qualità d'immagine predice la qualità geometrica?
   Si calcola la correlazione tra la classifica per PSNR corretto e quella per distanza mediana,
   ricordando che con 5 metodi il risultato è solo descrittivo.

## Fase 11 — README e materiali

1. **README, riga 317.** Nel comando di `app.paper` `--out paper\arco` contiene di nuovo un
   carattere di controllo (`\a` → BEL). Va corretto e va aggiunto `tests/test_docs.py`, che fallisce
   se `README.md`, `CHANGELOG.md` o i file `PIANO_*.md` contengono caratteri di controllo diversi
   da tab e a capo.
2. **Risultati e conclusioni** del README e dell'abstract si riscrivono **solo dopo** le fasi 7–9
   (e la 10, se fatta). Dove i valori grezzi e quelli corretti portano a conclusioni diverse, si
   riportano entrambi, e nelle conclusioni si dice quale dei due regge l'affermazione.
3. **Rigenerazione dei materiali**: `python -m app.paper --project foto --out paper\foto --scartate foto\LEGGIMI.txt`.
   Poi si comunicano all'utente i numeri che cambiano rispetto alla versione 0.4.0.

---

## Ordine consigliato e costo

| Fase | Training nuovi | Costo stimato |
|---|---|---|
| 7 | nessuno | una rivalutazione per run: il rendering a 20 MP è veloce per il 3DGS, circa 40 minuti per i NeRF su 111 viste |
| 8 | nessuno | si può fare nello stesso passaggio della fase 7 |
| 9 | 2 × nerfacto | circa 1,2 h più la valutazione |
| 10 | nessuno (MVS) | ore di calcolo per patch-match su 887 foto; va stimato su 40 foto |
| 11 | nessuno | minuti |

Si consiglia di implementare insieme le fasi 7 e 8, in modo che ogni run sia rivalutato una volta
sola con tutte le misure nuove.
