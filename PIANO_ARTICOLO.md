# Piano di lavoro per l'articolo — istruzioni per Claude Code

> Questo file va nella radice del repository `studio-gaussian-splatting-laboratorio-ictbc`.
> Chi lo legge (Claude Code) deve preparare il codice e gli script che servono all'articolo sul
> confronto tra 3D Gaussian Splatting, NeRF e fotogrammetria (caso di studio: Arco di Traiano,
> Benevento). Si parte dalla versione 0.3.1 (commit d8984b6).
>
> Si lavora per fasi, nell'ordine indicato. Al termine di ogni fase: test verdi
> (`.venv\Scripts\python -m pytest tests`), una voce nel `CHANGELOG.md` e un breve resoconto
> all'utente. Prima di modificare il codice delle fasi 1–2 si mostra il piano all'utente, perché
> queste fasi cambiano i risultati.

---

## 0. Contesto (perché si fa)

L'articolo sostiene che **il protocollo** è il contributo: stesse foto, stesse viste di test,
risoluzione piena e stesso codice di misura per cinque configurazioni (`inria-3dgs`, `splatfacto`,
`splatfacto-big`, `nerfacto`, `nerfacto-big`) più la fotogrammetria COLMAP. Il rilievo conta 887
foto su 906, da 5280×3956 px.

Domande di ricerca:

- **RQ1.** A parità di condizioni, come si collocano 3DGS e NeRF per fedeltà delle immagini,
  costo, dimensione e velocità di rendering?
- **RQ2.** Quanto pesa l'implementazione (Inria vs gsplat) rispetto alla famiglia di metodi?
  La risposta viene dai confronti appaiati vista per vista.
- **RQ3.** Quanto incidono le condizioni di presa? Le foto sono state scattate tra le 9:25 e le
  16:08, e una parte sono fotogrammi sviluppati da DNG.
- **RQ4** (solo se c'è un riferimento metrico). La qualità d'immagine predice la qualità geometrica?

Prima dei run definitivi vanno corrette due asimmetrie a favore o a sfavore dei NeRF. Sono già
verificate sul codice di nerfstudio 1.1.5:

- **A — ottimizzazione delle pose.** `nerfacto` e `nerfacto-big` hanno per default
  `camera_optimizer = CameraOptimizerConfig(mode="SO3xR3")`. Il correttore si applica solo in
  training (`nerfacto.py`, `get_outputs`: `if self.training: self.camera_optimizer.apply_to_raybundle`),
  mentre le viste di test sono rese con le pose COLMAP non corrette. `splatfacto` ha `mode="off"`
  e il codice Inria non ottimizza le pose. I NeRF risultano quindi penalizzati in valutazione.
- **B — compensazione dell'aspetto.** `nerfacto` ha `use_appearance_embedding = True` (un vettore
  per foto, dimensione 32) e sulle viste di test usa l'embedding medio
  (`use_average_appearance_embedding = True`). Assorbe così le differenze di luce e di sviluppo,
  cosa che il 3DGS non fa. La frase del README «Nessuno dei metodi configurati compensa le
  differenze di esposizione» non è quindi esatta.

---

## Fase 1 — Opzioni di protocollo e tracciabilità (cambia i risultati)

1. **Nuove impostazioni in `Settings` (`app/config.py`)**, salvate in `project.json`:
   - `pose_refinement: bool = False`: correzione delle pose nei metodi nerfstudio;
   - `appearance_embedding: bool = False`: embedding di aspetto per foto nei NeRF.

   I default sono quelli del confronto principale, cioè tutti i metodi alle stesse condizioni.
   Vanno esposti in «Opzioni avanzate» (`app/window.py`) e da riga di comando (`app/cli.py`:
   `--pose-refinement`, `--appearance-embedding`).

2. **Comando di training nerfstudio (`app/pipeline.py`, dove si costruisce `ns-train`).** Si
   aggiungono gli argomenti **prima** del sottocomando `nerfstudio-data`:
   - `--pipeline.model.camera-optimizer.mode` con valore `off` oppure `SO3xR3`, per tutti i
     metodi nerfstudio, così che anche splatfacto sia esplicito;
   - solo per la famiglia `nerf`: `--pipeline.model.use-appearance-embedding` `True|False`.

   Si verifica che le varianti `-disco` registrate in `app/ns_disk.py` accettino gli stessi
   argomenti, e che `ns-eval`/`eval_setup` ricarichino il modello senza errori con l'embedding
   disattivato.

3. **`run.json`**: si registrano `pose_refinement`, `appearance_embedding` e il picco di memoria
   grafica `gpu_peak_mb`. Per quest'ultima si interroga `nvidia-smi --query-gpu=memory.used
   --format=csv,noheader,nounits` ogni 2 s durante il passo di training, in un thread del runner
   (`app/runner.py`), e si salva il massimo. Il valore include gli altri processi sulla GPU: va
   scritto nel commento.

4. **Report (`app/report.py`, `collect`).** Oggi entrano nel confronto solo i run con stessa
   risoluzione e stesse iterazioni. Alle condizioni si aggiungono `pose_refinement` e
   `appearance_embedding`. I run vecchi senza questi campi valgono come `pose_refinement=True` e
   `appearance_embedding=True` per i NeRF, che erano i default di nerfstudio.

5. **Test (senza GPU)**, in `tests/test_pipeline.py`:
   - il comando `ns-train` contiene `camera-optimizer.mode off` per tutti i metodi nerfstudio;
   - contiene `use-appearance-embedding False` solo per i NeRF;
   - i valori opposti compaiono quando le opzioni sono attive;
   - il report non mescola run con condizioni diverse.

6. **README**:
   - correggere il limite sull'esposizione, spiegando le due opzioni e il default;
   - aggiungere al protocollo il punto «Pose e aspetto»;
   - correggere la riga 218, dove `exports\arco\...` contiene un carattere di controllo `\a`.
     Il testo giusto è `exports\arco\splatfacto_<data>\splat.ply`.

7. **CHANGELOG**: nuova versione 0.4.0, nella sezione «Cambiamenti che incidono sui risultati».

---

## Fase 2 — Verifica delle immagini di riferimento (limite C)

nerfstudio e il codice Inria preparano l'immagine vera di ogni vista con procedure di
correzione della distorsione diverse. Il limite va quantificato.

- Nuovo modulo `app/check_gt.py`, eseguibile nell'ambiente con PyTorch e GPU. Per ogni vista di
  test (da `split.json`) prende l'immagine vera da nerfstudio (`get_image_metrics_and_images` come
  in `app/evaluate.py`) e da Inria (`view.original_image`), le porta alla stessa dimensione
  **solo se differiscono** (segnalandolo) e calcola PSNR, SSIM e la differenza di dimensione.
  Le immagini si confrontano in memoria, mai dai JPG salvati in `test_views/`, che sono compressi.
- L'uscita è `reports/<progetto>/check_gt.json`, con le misure per vista e le medie.
- Come criterio di lettura per l'articolo: se il PSNR medio supera i 40 dB, basta citarlo come
  limite trascurabile. Altrimenti si avvisa l'utente, perché va deciso se valutare tutti i metodi
  sulle stesse immagini.

---

## Fase 3 — Script di analisi per l'articolo (non cambia i risultati)

Nuovo modulo `app/paper.py`, eseguito con
`.venv\Scripts\python -m app.paper --project arco --out paper\arco`. Legge solo i file già
prodotti (`run.json`, `metrics.json`, `photos.json`, `alignment.json`, `split.json`, `mesh.json`,
modello COLMAP) e scrive in `paper/<progetto>/` le uscite seguenti. Ogni tabella si scrive sia
in CSV sia in LaTeX (booktabs).

| Uscita | Contenuto |
|---|---|
| `tab1_dati.*` | foto totali, per camera (JPG/DNG), scartate per motivo, usate, registrate, training/test, errore medio di riproiezione, punti 3D, lunghezza media delle tracce |
| `tab2_configurazioni.*` | metodo, famiglia, implementazione, iterazioni, fattore di riduzione e lato in px, `pose_refinement`, `appearance_embedding`, `photos_in` |
| `tab3_risultati.*` | PSNR, SSIM, LPIPS (media ± dev. st. e IC 95% bootstrap con 10.000 ricampionamenti sulle viste), training in ore, `gpu_peak_mb`, fps, dimensione del `.ply` in MB, numero di gaussiane |
| `tab4_appaiati.*` | per ogni coppia di metodi e metrica: differenza mediana vista per vista, IC 95% bootstrap, test di Wilcoxon appaiato, p corretto con Holm |
| `tab5_fotogrammetria.*` | tempi per passo (da `mesh.json`), punti di `fused.ply`, vertici e triangoli della mesh |
| `fig2_camere.pdf/.png` | centri delle camere visti dall'alto; test evidenziate, JPG e DNG con colori diversi |
| `fig5_ora.pdf/.png` | PSNR (e LPIPS) per vista rispetto all'ora di scatto (EXIF `taken` in `photos.json`, formato `YYYY:MM:DD HH:MM:SS`), un pannello per metodo |
| `fig5b_jpg_dng.pdf/.png` | distribuzione delle metriche per vista, strato JPG vs DNG, per metodo |
| `numeri.json` | tutti i numeri citati nel testo, con chiavi stabili (es. `inria-3dgs.psnr_mean`), per riempire l'abstract senza ricopiature a mano |

Note di implementazione:

- Per l'articolo si considerano solo i run confrontabili: stesso allineamento
  (`runs.alignment_id`) e stesse condizioni (fase 1, punto 4). Con più run per metodo (semi
  diversi, fase 4) si riportano media e dev. st. tra run, e i confronti appaiati si fanno sulla
  media per vista tra i semi.
- La camera di una vista si ricava dalla sottocartella del nome (`jpg/…` o `dng/…`).
- Per le pose serve leggere da `images.bin` anche quaternione e traslazione, oggi scartati in
  `app/colmap_model.read_images`. Si aggiunge una funzione separata, senza cambiare quella esistente.
- Le figure riprendono lo stile di `app/report.py` (`FAMILY_COLOR`, `_style`) e si salvano in PDF
  vettoriale e in PNG a 300 dpi.
- `scipy` serve per Wilcoxon: va aggiunto a `requirements.txt`, con import solo dentro
  `app/paper.py`.
- Test in `tests/test_paper.py`, con run finti costruiti da `conftest.py`: tabelle, Holm,
  bootstrap con seme fisso e abbinamento delle viste tra metodi.

---

## Fase 4 — Variabilità tra esecuzioni (limite D)

- Si aggiunge `--seed N` a `cli run`/`cli test` e si registra il seme in `run.json`.
  - nerfstudio: `--machine.seed N`.
  - Codice Inria: il seme è fisso a 0 in `utils/general_utils.safe_state`. Va verificato sul
    commit fissato e poi impostato tramite variabile d'ambiente letta da un piccolo wrapper,
    analogo a `app/inria_lazy.py`, **senza modificare il codice Inria**. Se non è praticabile, si
    documenta che la variabilità del metodo Inria viene solo dal non-determinismo della GPU.
- Il report e `app/paper.py` raggruppano i run che differiscono solo per il seme.

---

## Fase 5 — Scala metrica (facoltativa, limite E)

Solo se l'utente conferma. Serve a dare una scala in metri ai modelli usando il GPS EXIF del
drone, ma **senza toccare l'allineamento condiviso**, il cui hash invaliderebbe tutti i run.

- Comando su una **copia** del modello sparso:

  ```
  colmap model_aligner --input_path colmap/sparse/<n> --output_path colmap/sparse_geo
      --database_path <database.db> --ref_is_gps 1 --alignment_type enu --alignment_max_error 3.0
  ```

  Prima va verificato che il database abbia le posizioni GPS dalle EXIF.
- Il fattore di scala e la trasformazione si salvano in `data/<progetto>/scala.json`. Si usano
  solo in `app/paper.py` per riportare dimensioni in metri e confrontare la nuvola densa con i
  centri delle gaussiane.

---

## Fase 6 — Piano dei run definitivi (da lanciare dopo le fasi 1–4)

Comandi dalla cartella del programma:

```powershell
# confronto principale: tutti i metodi, risoluzione piena, pose e aspetto disattivati (default)
.venv\Scripts\python -m app.cli test --project arco --photos D:\foto\arco

# NeRF con le opzioni originali di nerfstudio, per discutere l'effetto (RQ3 / tab. 2)
.venv\Scripts\python -m app.cli test --project arco --methods nerfacto --pose-refinement --appearance-embedding

# sensibilita' alla risoluzione per i due 3DGS (risponde a Fadilah et al. 2026)
.venv\Scripts\python -m app.cli test --project arco --methods splatfacto inria-3dgs --quality alta
.venv\Scripts\python -m app.cli test --project arco --methods splatfacto inria-3dgs --quality media

# variabilita' tra esecuzioni (3 semi, qualita' media)
.venv\Scripts\python -m app.cli test --project arco --methods splatfacto inria-3dgs --quality media --seed 1
.venv\Scripts\python -m app.cli test --project arco --methods splatfacto inria-3dgs --quality media --seed 2

# verifica delle immagini di riferimento e materiali dell'articolo
.venv\Scripts\python -m app.check_gt --project arco
.venv\Scripts\python -m app.paper --project arco --out paper\arco
```

Prima di lanciare, si stimano i tempi dal set di prova da 40 foto e si comunicano all'utente. Il
confronto a risoluzione piena con 887 foto può richiedere giorni di calcolo.

---

## Regole

- Lo stile del codice resta quello del repository: commenti e messaggi in italiano, accenti
  scritti con l'apostrofo nei commenti del codice, nessun cambiamento silenzioso dei default che
  incidono sui risultati, e ogni scelta registrata in `run.json`.
- I test non devono richiedere la GPU.
- Non si modifica il codice Inria né i pacchetti installati: si interviene solo con argomenti da
  riga di comando o con wrapper come `inria_lazy.py`.
- Per decisioni di metodo non coperte da questo file (per esempio una soglia o un default nuovo)
  si chiede all'utente.
