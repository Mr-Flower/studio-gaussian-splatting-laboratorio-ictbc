# Piano di lavoro per l'articolo — quarta parte: riferimento Metashape

> Va nella radice del repository, accanto ai piani precedenti, e ne continua la numerazione e le
> regole: test senza GPU, una voce nel `CHANGELOG.md` per fase, nessuna modifica al codice Inria,
> una domanda all'utente per le scelte di metodo non scritte qui. Le fasi 12 e 13 del terzo piano
> restano valide. La fase 16 riusa la procedura dei punti di superficie della fase 12.

---

## Contesto

L'utente ha elaborato il rilievo anche con **Agisoft Metashape Professional 2.3.0** (report del
27/10/2026). Riepilogo:

| Voce | Valore |
|---|---|
| Immagini | 1983, di cui 1856 allineate |
| Camera drone L2D-20c, 5272 × 3948 | 907 immagini: i DNG di tutti gli scatti, con distorsione forte (K1 −0,10) |
| Camera drone L2D-20c, 5280 × 3956 | 853 immagini: i JPG, già corretti dalla camera (K1 +0,011) |
| Camera a terra Sony ZV-1F, 5472 × 3648 | 220 immagini, **assenti dal confronto dei metodi** |
| Errore di riproiezione | 0,957 px |
| Nuvola densa | 449.432.795 punti, mappe di profondità «High», filtro «Mild» |
| Mesh | 47,1 milioni di facce |
| Georeferenziazione | solo GPS delle camere (WGS 84), nessun punto di controllo a terra. Errore medio delle posizioni: X 2,97 m, Y 2,19 m, **Z 10,50 m** |
| Macchina | Windows 64 bit, i9-13900K, 63,7 GB di RAM, NVIDIA RTX A6000 |

Questa nuvola sarà il **riferimento geometrico esterno**, per tre ragioni: è calcolata da un altro
software, a risoluzione più alta (mappe a «High», mentre COLMAP lavorava a 1600 px) e usa anche le
foto da terra. In questo modo la fotogrammetria di COLMAP smette di essere il riferimento e diventa
**uno dei metodi confrontati**, e l'articolo confronta davvero le tre famiglie. Da dichiarare
nell'articolo:

1. il riferimento **non è indipendente** dalle foto del drone (le usa anche lui) e **non è metrico**:
   la scala viene dal GPS, la cui quota sbaglia di circa 10 m. La scala resta quella di
   `scala.json`, da verificare con la fase 13;
2. contiene zone viste **solo dalle foto a terra**: la completezza dei metodi va misurata solo sulla
   parte visibile dal drone (punto 4 della fase 16).

---

## Fase 16 — La nuvola Metashape come riferimento

### Cosa fa l'utente in Metashape

I nomi dei menu possono variare leggermente con la versione. Prima di procedere, Claude Code chiede
all'utente i file esportati e la loro posizione.

1. **Sistema di coordinate metrico.** Nel pannello Reference impostare WGS 84 / UTM zona 33N
   (EPSG:32633). Le coordinate geografiche in gradi non servono.
2. **Posizioni stimate delle camere.** Pannello Reference → vista «Estimated» → *Export Reference*
   in CSV, con nome dell'immagine e X, Y, Z stimati in UTM 33N. Come riserva, anche
   *File → Export → Export Cameras* nel formato Agisoft XML.
3. **Nuvola densa.** *File → Export → Export Point Cloud* in PLY binario, UTM 33N, con colori e
   normali. Il file supera i 10 GB: va messo su un disco con spazio.

### Cosa fa Claude Code

1. **Lettura** (`app/reference.py`, nuovo). Il PLY da 449 milioni di punti si legge a blocchi
   (memmap su `plyfile` o lettura binaria diretta), con sottocampionamento a voxel di 5 mm in
   coordinate UTM. Si scrive `data/<progetto>/riferimento/metashape_5mm.ply` con il numero di punti
   prima e dopo. Le coordinate UTM vanno traslate prima della conversione in float32, per non
   perdere precisione.
2. **Registrazione nel sistema di COLMAP.**
   - Si abbinano per nome le camere comuni: gli 853 JPG ci sono in entrambi i progetti. I nomi in
     Metashape possono non avere il percorso; si abbinano sul nome del file. Si verifica che
     coincidano con le foto registrate da COLMAP.
   - Si stima una similitudine 3D (Umeyama, con scala) dai centri Metashape ai centri COLMAP, con
     scarto iterativo dei residui oltre 3 volte la mediana, come in `scala.py`.
   - Si rifinisce con un ICP punto-piano tra la nuvola Metashape sottocampionata e la nuvola densa
     COLMAP, limitato al riquadro di `geometry.region`. Va implementato con `cKDTree` e minimi
     quadrati, senza Open3D. Si accetta solo se la variazione di scala resta sotto lo 0,5%.
   - In `riferimento/registrazione.json` si salvano: trasformazione, numero di camere usate,
     residui dei centri (mediana e 90° percentile, in unità COLMAP e in metri), correzione portata
     dall'ICP e distanza mediana finale tra le due nuvole.
   - **Controllo**: il fattore di scala tra le due ricostruzioni, letto come metri per unità COLMAP
     (Metashape è in metri UTM), va confrontato con le due stime di `scala.json` (2,49 e 2,79) e
     riportato. È una terza stima GPS, **non** una verifica indipendente: va scritto così.
3. **Il riferimento nel confronto geometrico** (`app/geometry.py`).
   - Si aggiunge l'opzione `--riferimento metashape|colmap`; il default diventa `metashape` quando
     esiste `riferimento/`.
   - Con `metashape` si confrontano sei metodi: i cinque radiance field, con i punti di superficie
     della fase 12, più la **nuvola densa COLMAP**, che diventa uno dei metodi.
   - Tabella `tab8_geometria`: una riga per metodo, con accuratezza e completezza come oggi.
     Tabella `tab8b_geometria_colmap`: il vecchio confronto con COLMAP come riferimento, da tenere
     come analisi secondaria.
4. **Completezza solo dove vede il drone.** Un punto Metashape si dice «visto dal drone» se in
   almeno 2 delle 887 camere COLMAP verifica tre condizioni: cade nell'immagine, è davanti alla
   camera ed è coerente con la profondità della mappa geometrica di COLMAP
   (`colmap/dense/stereo/depth_maps/*.geometric.bin`), con tolleranza pari a 3 cm convertiti in
   unità COLMAP con `scala.json`.
   - La completezza si calcola solo su questi punti. Si riporta la frazione di punti Metashape
     esclusi, cioè quelli visti solo da terra.
   - L'accuratezza (metodo → riferimento) usa tutto il riferimento.
   - Test senza GPU: proiezione, lettura delle mappe di profondità binarie di COLMAP (formato
     `width&height&channels&` seguito da float32) e conteggio delle viste su una scena sintetica.
5. **GSD del rilievo da drone**, per la sezione sul caso di studio. Il valore di 1,46 mm/pixel del
   report Metashape è una media che include le foto da terra. Si calcola invece, per ogni immagine
   del drone, il GSD = distanza mediana dalla camera ai suoi punti 3D della nuvola sparsa × dimensione
   del pixel (3,36 µm) / focale (12,29 mm), convertito in metri con `scala.json`. Vanno in
   `numeri.json` la mediana e il 10°–90° percentile, oltre alla distanza mediana camera–monumento.
6. **Figure**: `fig6c_profilo` della fase 12 con la nuvola Metashape in grigio come riferimento, e
   una figura con la mappa delle zone viste solo da terra, in pianta e in prospetto.
7. Nel README: aggiornare risultati e limiti del confronto geometrico, la descrizione del
   riferimento e i dati del rilievo (numero di scatti, camera a terra).

## Fase 17 — Dati del rilievo nei materiali

1. Il report Metashape conta **907 DNG**, uno per scatto. Il rilievo ha quindi 907 scatti: il caso di
   studio parte da 906 perché lo scatto 636, una foto di gruppo, è stato tolto a mano
   (`scripts/0_prepare_arco.py`). Va confermato con l'utente e scritto nel README e in
   `tab1_dati`: 907 scatti, 1 tolto a mano, 906 analizzati, 19 scartati, 887 usati.
2. In `tab1_dati` si aggiunge la camera a terra Sony ZV-1F (220 foto, 5472 × 3648), indicando che
   è usata solo dal riferimento Metashape.
3. Hardware nel README e in `numeri.json`: si chiede all'utente se la macchina del report Metashape
   (i9-13900K, 63,7 GB di RAM) è la stessa usata per i training. In caso affermativo si registrano CPU
   e RAM. Conviene anche aggiungere a `run.json` la lettura automatica di CPU e RAM per i run futuri
   (`platform`, `psutil`).

---

## Ordine e costo

| Fase | Costo stimato |
|---|---|
| 16.1 lettura e sottocampionamento | tempo dominato dalla lettura di oltre 10 GB: decine di minuti |
| 16.2 registrazione | minuti |
| 16.3–16.4 confronto e visibilità | da minuti a un'ora (proiezione su 887 mappe di profondità) |
| 16.5–16.7 GSD, figure, README | minuti |
| 17 | minuti, più le risposte dell'utente |

Le fasi 12 e 13 restano la priorità: la 16 usa i punti di superficie della 12, e la scala in metri
dipende dalla 13.
