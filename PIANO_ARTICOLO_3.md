# Piano di lavoro per l'articolo — terza parte (dopo la versione 0.5.0)

> Va nella radice del repository, accanto a `PIANO_ARTICOLO.md` e `PIANO_ARTICOLO_2.md`, e ne
> continua la numerazione. Le regole sono le stesse: test verdi senza GPU, una voce nel
> `CHANGELOG.md` per fase, nessuna modifica al codice Inria o ai pacchetti installati, e una
> domanda all'utente per le scelte di metodo non scritte qui.
>
> Nessuna di queste fasi richiede nuovi training, salvo la 14, che è facoltativa. Le fasi 12 e 15
> cambiano affermazioni dell'abstract: non si riscrivono risultati e conclusioni del README prima
> che la fase 12 sia finita.

---

## Perché

La versione 0.5.0 regge su quasi tutto. Restano tre punti che un revisore attaccherebbe.

1. **Il confronto geometrico non è alla pari, e su di esso poggia una conclusione dell'abstract.**
   L'abstract dice che i NeRF hanno una geometria più precisa e che la qualità delle immagini non
   predice l'accuratezza. Ma le nuvole confrontate sono di natura diversa:
   - per i NeRF, `ns-export pointcloud` campiona punti **sulla superficie stimata**: la profondità
     attesa lungo raggi delle foto di training. Toglie poi gli outlier (`remove_outliers=True`,
     `std_ratio=10` per default in nerfstudio 1.1.5) e produce 1 milione di punti;
   - per il 3DGS si usano i **centri delle gaussiane** con opacità > 0,5: stanno dentro o attorno
     alla superficie, nessun filtro, e un numero di punti diverso da metodo a metodo.

   La coda del 3DGS (90° percentile 29–37 cm) e la completezza, che dipende dalla densità dei punti,
   possono venire dalla rappresentazione scelta per il confronto e non dal metodo. Il limite è
   dichiarato nel README, ma va rimosso: ricavare i punti di superficie **allo stesso modo per
   tutti i metodi**.
2. **«Allenare a risoluzione piena non conviene» è scritto senza condizioni nell'abstract.** Vale
   con 30.000 iterazioni e i parametri di densificazione predefiniti: il numero di gaussiane non
   cresce con la risoluzione (Inria 0,73 milioni a 5280 px contro 1,15 a 1320 px). La condizione
   va scritta anche nell'abstract. Se si vuole, la si mette alla prova con un run (fase 14).
3. **La scala non è verificata.** Le due stime dal GPS (2,49 e 2,79 m per unità) differiscono
   dell'11%, e tutte le distanze in centimetri dipendono da quella scelta.

---

## Fase 12 — Punti di superficie con la stessa procedura per tutti (solo rendering)

1. **Nuovo modulo `app/surface.py`**, eseguito nell'ambiente con PyTorch:
   `python -m app.surface --project foto`. Per ogni run del confronto principale (tre 3DGS e due
   NeRF):
   - **Camere**: una vista di training ogni 8 (circa 97), in ordine di nome, sempre le stesse per
     tutti i metodi, rese con lato maggiore 1320 px. Ogni metodo usa le **proprie** camere
     (nerfstudio con la distorsione OPENCV, Inria con le camere già raddrizzate), così come in
     valutazione.
   - **Per pixel**: profondità attesa e opacità accumulata. Si tengono i pixel con opacità > 0,5,
     si ritroiettano in 3D e si portano nel sistema di COLMAP (`clean.to_colmap`).
     - nerfstudio (splatfacto e nerfacto): `pipeline.model.get_outputs_for_camera(camera)` dà
       `depth` e `accumulation`. **Attenzione alle convenzioni**: in splatfacto `depth` è la
       profondità lungo l'asse z della camera, in nerfacto è la distanza lungo il raggio
       (l'esportatore di nerfstudio fa `origins + directions * depth`). Va scritto un test che
       verifichi la convenzione: ritroiettare i punti nella camera deve restituire il pixel di
       partenza.
     - codice Inria: si controlla sul commit fissato se `render()` restituisce la profondità o la
       sua inversa. Se non è chiaro, si usa il metodo indipendente dalla versione: si rende con
       `override_color` = z in camera di ogni gaussiana (ripetuto sui tre canali) per avere la z
       pesata, e con `override_color` = 1 per l'opacità accumulata. La profondità è il rapporto
       dove l'opacità supera 0,5.
   - **Campionamento**: si uniscono i punti di tutte le viste, si ritagliano al riquadro di
     `geometry.region` e si estraggono **1.000.000 punti con seme 0**.
   - **Filtro**: **nessuno**, uguale per tutti. Per confronto si calcola anche la variante con lo
     stesso filtro statistico per tutti (vicini = 20, `std_ratio` = 10, come nerfstudio),
     implementato con `cKDTree` per non aggiungere Open3D.
   - Uscita: `exports/<progetto>/<metodo>_<data>/superficie.ply` e `superficie.json` (viste
     usate, pixel validi, punti prima e dopo il campionamento).
2. **Controllo di coerenza per i NeRF**: si riesporta nerfacto con
   `ns-export pointcloud ... --remove-outliers False` e si confronta con `superficie.ply` dello
   stesso run. Le due nuvole devono avere distanza mediana reciproca ben sotto il centimetro;
   se non è così, ci si ferma e si avvisa l'utente.
3. **`app/geometry.py`**:
   - il confronto principale usa `superficie.ply` per **tutti** i metodi;
   - i centri delle gaussiane restano come analisi secondaria (colonna «centri»), con lo stesso
     sottocampionamento;
   - `tab8_geometria` riporta per ogni metodo: superficie (accuratezza e completezza), superficie
     filtrata e centri;
   - la ρ di RQ4 si calcola sulla superficie non filtrata, e in `numeri.json` si scrive
     esplicitamente che con 5 metodi il risultato è solo descrittivo.
4. **Figura `fig6c_profilo`**: una sezione verticale sottile (spessore 5 cm) attraverso il fornice,
   con i punti della nuvola densa in grigio e quelli di superficie di ciascun metodo in colore, un
   pannello per metodo. È la figura che mostra a un lettore dei beni culturali che cosa significano
   i centimetri.
5. Test senza GPU: ritroiezione e retroproiezione su una camera sintetica con un piano noto, nelle
   due convenzioni di profondità; filtro statistico su una nuvola con outlier noti.

## Fase 13 — Verifica della scala con una dimensione del monumento

1. In `app/scala.py`, nuova funzione `misura_arco`:
   - dalla nuvola densa nel sistema raddrizzato (`geometry.upright_frame`) si stima il suolo come
     2° percentile delle quote dei punti vicini all'arco;
   - si tengono i punti tra 1 m e 6 m sopra il suolo, una fascia che taglia i piloni e non le
     strade, ritagliati al riquadro di `geometry.region`;
   - si calcola il rettangolo di area minima della loro pianta (assi principali, più una ricerca
     sull'angolo a passi di 0,5°), e la **lunghezza del lato maggiore** è la larghezza della
     facciata;
   - si calcola anche l'**altezza**: 99,5° percentile delle quote meno il suolo.
2. Valori di confronto: **larghezza 8,60 m, altezza 15,60 m**, riportati da italia.it, il portale
   del turismo, quindi da una fonte divulgativa. Si scrivono in `scala.json` (`verifica`) con la
   fonte, insieme alle dimensioni misurate con le due scale (2,49 e 2,79 m per unità) e allo scarto
   percentuale di ciascuna.
3. Si comunica all'utente quale scala è coerente. Se nessuna delle due resta entro il 3% sulla
   larghezza, ci si ferma: serve una misura migliore, dal rilievo del 2017 o una misura diretta.
   L'altezza dipende dalla quota del suolo davanti all'arco ed è solo un controllo secondario.

## Fase 14 (facoltativa) — La risoluzione piena con più capacità

Va fatta solo se l'utente vuole sostenere l'affermazione sulla risoluzione oltre la condizione
«parametri predefiniti».

1. Un run di splatfacto a qualità massima (5280 px) con soglia di densificazione dimezzata:
   `--pipeline.model.densify-grad-thresh 0.0004` (default 0.0008, con `use_absgrad` già attivo).
   Va aggiunta un'opzione `--densify-grad-thresh` in `cli` e il valore va registrato in `run.json`,
   perché fa parte delle condizioni.
2. Si valuta come gli altri (grezzo, corretto, a 2640 e 1320 px) e si riportano in
   `tab6_risoluzione` come riga a parte: numero di gaussiane, picco di memoria, training.
3. Lettura per il README: se a 1320 px il run raggiunge o supera quello allenato a 1320 px, la frase
   diventa «a parità di parametri predefiniti». Altrimenti l'affermazione si rafforza. Il costo
   stimato è di circa 3 ore.

## Fase 15 — Affermazioni dell'abstract e del README

Da fare dopo le fasi 12 e 13 (e la 14, se fatta). Si riscrivono solo le frasi che seguono, poi si
mostrano all'utente le versioni nuove accanto alle vecchie.

1. **Risoluzione**: «Allenare a risoluzione piena non conviene» diventa «Con 30.000 iterazioni e
   i parametri di densificazione predefiniti, allenare a risoluzione piena non conviene», da
   aggiornare con l'esito della fase 14.
2. **Geometria**: le frasi su accuratezza, completezza e RQ4 si riscrivono con i numeri della
   superficie. «La qualità delle immagini non predice l'accuratezza geometrica» resta solo se
   regge sui punti di superficie. In ogni caso va formulata come osservazione descrittiva su 5
   metodi e rispetto alla nuvola densa calcolata a 1600 px, che è un riferimento e non una verità.
3. **Esposizione**: dopo la correzione affine resta una correlazione negativa. Nel testo va
   indicata l'interpretazione più probabile: a esposizioni diverse corrispondono condizioni di
   luce diverse (ombre, posizione del sole nell'arco della giornata) che nessuna correzione globale
   può compensare. Per sostenerla si calcola anche la ρ tra PSNR corretto e ora di scatto, e quella
   tra |ΔEV| e ora, e si scrivono in `numeri.json`.
4. **Limiti**: il punto sul confronto geometrico non alla pari si sostituisce con la descrizione
   della procedura unica della fase 12. Restano il riferimento a 1600 px e l'assenza di misure
   indipendenti, e si aggiunge la verifica della scala con la sua fonte.
5. Si rigenerano i materiali (`python -m app.paper ...`) e il sito, e si comunicano all'utente i
   numeri cambiati rispetto alla 0.5.0.

---

## Ordine e costo

| Fase | Training | Costo stimato |
|---|---|---|
| 12 | no | rendering di circa 97 viste a 1320 px per 5 modelli: pochi minuti per il 3DGS, qualche minuto anche per i NeRF (circa 1,3 milioni di raggi per vista) |
| 13 | no | minuti |
| 14 | 1 × splatfacto a 5280 px | circa 3 h più la valutazione |
| 15 | no | minuti, più la revisione dell'utente |
