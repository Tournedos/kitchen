# DEXTER kinematics – pipeline ATHENA per il tracking 3D delle mani

*Versione inglese: [README.md](README.md)*

Pipeline per ricostruire in 3D mani (e corpo) da registrazioni multicamera (9 Basler, `multicam_capture`)
usando [ATHENA](https://github.com/neural-control-and-computation-lab/athena) (MediaPipe + triangolazione),
con calibrazione da scacchiera e analisi della qualità dei dati estratti.

```
scripts/
├── athena_calibration.py   # 1. calibrazione multicamera dalla scacchiera -> formato ATHENA
├── athena_analysis.py      # 2. pipeline ATHENA headless + grafici di qualità + log dei tempi (singola / batch)
├── collect_data3d.py       #    raccoglie tutti i video data3d.mp4 in videos3ddata/
├── hand_angles.py          #    angoli articolari (MCP, PIP, DIP, TAM, spread, pollice, polso, avambraccio)
├── athena_gui.py           #    (opzionale) GUI originale di ATHENA con font ingranditi
└── requirements.txt
notebooks/
└── landmarks3d_viewer.ipynb  # 3. visualizzazione interattiva dei tracciati 3D e della loro qualità
data/                       # registrazioni (una cartella per registrazione)
videos3ddata/               # video dello scheletro 3D, uno per registrazione (collect_data3d.py)
exports/                    # grafici (HTML) e dati (CSV) salvati dal notebook
diario/                     # diario di lavoro
athena_runs_log.csv         # log di tutte le esecuzioni (creato automaticamente)
.vscode/settings.json       # VSCode: interprete conda + download dei widget per il notebook
```

---

## 1. Installazione (una volta sola)

ATHENA richiede **Python 3.12**. Si usa un ambiente conda.

```bash
# Miniforge (se conda non è installato)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash && source ~/.bashrc

# Ambiente
conda create -n athena python=3.12
conda activate athena
pip install athena-tracking opencv-python
pip install jupyterlab ipywidgets plotly pandas ipykernel "numpy==1.26.4"   # notebook di visualizzazione
python -m ipykernel install --user --name athena --display-name "Python (athena)"
pip install "mediapipe==0.10.21" "moviepy<2"     # versioni compatibili con ATHENA 0.9 (vedi sotto)
conda install -c conda-forge ffmpeg
```

> **Perché le versioni fissate**: `mediapipe` ≥ 0.10.30 ha rimosso `mediapipe.solutions` (errore
> *cannot import name 'solutions'*), `moviepy` 2.x ha rimosso `moviepy.editor` (usato da `athena-montage`).
> pip segnala conflitti di numpy con `astropy`/`tifffile`/`opencv-python 5`: si possono ignorare, ATHENA non li usa.

Verifica:
```bash
conda activate athena
python -c "import athena.labels2d, athena.triangulaterefine, athena.montage; print('ok')"
```

Tutti i comandi seguenti vanno eseguiti con l'ambiente attivo (`conda activate athena`) dalla cartella del progetto
`~/Desktop/DEXTER_kinematics`.

---

## 2. Struttura dei dati

Una registrazione di `multicam_capture` (`data/<registrazione>/`) contiene:

```
data/2026-11-10_112048_louis_finertapping_01/
├── Take_<stamp>-M<seriale>.mkv          # un video per camera (MJPEG, 1920x1200, 40 fps)
├── Metadata_<stamp>-M<seriale>.json     # timestamp PTP di ogni frame (usati per il QC acquisizione)
├── metadata.json, recording_verification.json, ...
└── calibration_frames/<cattura>/
    ├── cam_<N>_frame<K>.png             # screenshot sincronizzati della scacchiera
    └── camera_config.json               # nome camera <-> seriale, focale
```

Gli script creano la *main folder* di ATHENA dentro la registrazione:

```
data/<registrazione>/athena/
├── calibration/cam_1.yaml ... cam_9.yaml      # da athena_calibration.py
├── calibration_check/cam_N.jpg                # controllo visivo calibrazione
├── calibration_report.json                    # errori, mapping camera <-> seriale <-> video
├── videos/<trial>/cam_1.mp4 ... cam_9.mp4     # link ai Take_*.mkv (0 byte), nomi/ordine per ATHENA
├── landmarks/<trial>/                         # dati ATHENA (2D per camera, 3D, 2D riproiettati)
├── videos_processed/<trial>/data3d.mp4        # scheletro 3D (sempre)
├── videos_processed/<trial>/camN[_refined].mp4  # overlay 2D / 3D riproiettato (profilo standard e full)
├── images/<trial>/, imagesrefined/<trial>/    # immagini per frame: solo con --outputs full / --keep-frames
└── analysis/<trial>/                          # export + grafici di qualità + run_info.json
```

`<trial>` è il nome della registrazione senza data/ora (es. `louis_finertapping_01`), oppure il nome
delle sottocartelle già presenti in `athena/videos/`.

**Raccomandazioni per le registrazioni** (dalle prime acquisizioni):
* avviare la registrazione **un paio di secondi prima** che inizi il movimento e fermarla **un paio di secondi
  dopo** la fine: nei primi e negli ultimi frame le mani sono spesso parzialmente fuori campo o non ancora
  rilevate, e lo smoothing è meno affidabile ai bordi del trial;
* evitare per quanto possibile occlusioni complete di una mano (corpo, altra mano, oggetti): una mano occlusa
  diventa *missing* (i buchi di 6 o più frame non vengono riempiti);
* le mani dovrebbero essere viste da **almeno 3 camere** per la maggior parte del tempo (controllare
  `qc_summary.json` o il notebook: con 2 camere una detection sbagliata non si può riconoscere).

---

## 3. Calibrazione – `athena_calibration.py`

Calcola intrinseci ed estrinseci delle camere dagli screenshot della scacchiera e scrive i file nel formato
letto da ATHENA.

```bash
python scripts/athena_calibration.py data/<registrazione>                    # scacchiera del lab, quadrati 100 mm
python scripts/athena_calibration.py data/<registrazione> --square-mm 80     # altra scacchiera
```

**`--square-mm`** (default **100**, la scacchiera del laboratorio ha quadrati da 10 cm): lato *reale* di un quadrato
della scacchiera, in mm. Va cambiato solo se si usa un'altra scacchiera.
Determina la scala di tutte le misure 3D: un valore sbagliato scala proporzionalmente tutte le distanze.

| Opzione | Default | Significato |
|---|---|---|
| `--inner-corners` | `4x6` | angoli interni (quadrati − 1 per lato); 4x6 = scacchiera 5x7 |
| `--format` | `yaml` | `yaml` (JARVIS, un file per camera) o `toml` (Anipose) |
| `--out` | `<registrazione>/athena` | main folder di output |
| `--origin` | `corner` | origine del sistema mondo: angolo vicino al bordo rosso, o `center` |
| `--intrinsics-mode` | `focal` | `focal`: stima solo la focale; `focal_k1`: + distorsione radiale; `nominal`: focale da lente/pixel; `full`: richiede molte pose |
| `--intrinsics-dir` | – | intrinseci già calibrati (più precisi): calcola solo gli estrinseci |
| `--pixel-size-um` | `5.86` | pixel pitch del sensore (Basler acA1920-40gc) |
| `--max-focal-deviation` | `0.15` | se la focale stimata devia più del 15% dalla nominale usa la nominale |
| `--flip-cams` | – | es. `3,7`: camere con assi ruotati di 180° nei check (bordo rosso letto male) |
| `--ref-frame`, `--frames-dir` | auto | scelta di frame/cartella screenshot |
| `--skip-video-check` | off | non confronta i video con gli screenshot |

**Sistema di riferimento** (mm): origine sulla scacchiera appoggiata sul tavolo, X lungo il lato lungo
(parallelo al bordo rosso), Y lungo il lato corto, **Z verso l'alto**.

**Controllo**: aprire `athena/calibration_check/cam_N.jpg` (assi devono essere coerenti tra le camere)
e leggere gli errori in `athena/calibration_report.json` (`triangulation_rms_mm`).

> Con un solo frame per camera gli intrinseci sono stime: per la massima accuratezza calibrare gli
> intrinseci una volta con molte pose della scacchiera e passarli con `--intrinsics-dir`.

---

## 4. Tracking e analisi – `athena_analysis.py`

Esegue in sequenza, senza GUI:

1. **prepare** – collega i `Take_*.mkv` come `athena/videos/<trial>/cam_N.mp4` (rinominati e ordinati come le
   calibrazioni; ATHENA abbina video e calibrazioni in ordine alfabetico). Nessuna copia: link simbolici.
2. **ATHENA 2D** – MediaPipe (mani + posa) su tutte le camere in parallelo (GPU).
3. **ATHENA 3D** – correzione destra/sinistra, triangolazione, smoothing, riproiezione, immagini e video.
4. **export** – CSV e npy dei landmark 3D.
5. **QC** – grafici e statistiche di qualità.
6. **log** – tempi di esecuzione e info della run in `athena_runs_log.csv`.

### Preparazione video (automatica)

ATHENA cerca solo file `.avi`/`.mp4` e abbina video e calibrazioni in ordine alfabetico, quindi i
`Take_<stamp>-M<seriale>.mkv` vanno rinominati. Lo fa il primo passo dello script (`prepare_videos`), a ogni
esecuzione:

* mapping seriale → `cam_N` letto da `athena/calibration_report.json` (sezione `athena_order`, scritta da
  `athena_calibration.py`): per questo la calibrazione va fatta prima;
* `athena/videos/<trial>/cam_N.mp4` è un **link simbolico relativo** al `Take_*.mkv` originale: nessuno spazio
  occupato, nessuna perdita (OpenCV riconosce il contenitore mkv indipendentemente dall'estensione).
  Il link è relativo, quindi resta valido se si sposta/copia l'intera cartella della registrazione;
* `--video-link hardlink` usa hard link (stesso disco), `--video-link copy` il vecchio comportamento
  (`ffmpeg -c copy` in un mp4, raddoppia lo spazio dei video);
* i video già presenti in `athena/videos/<trial>/` non vengono rifatti; se la cartella viene cancellata (o i
  link si rompono, es. copia su un disco FAT/exFAT che non supporta i link) viene ricreata alla run successiva;
* controlla che il numero di video sia uguale al numero di file di calibrazione e che tutti abbiano lo stesso
  numero di frame (warning altrimenti).

### Uso

```bash
# test veloce sul 10% dei frame
python scripts/athena_analysis.py data/<registrazione> --fraction 0.1

# esecuzione completa (profilo di output standard)
python scripts/athena_analysis.py data/<registrazione>

# senza argomento: si apre una finestra per scegliere la cartella
python scripts/athena_analysis.py

# solo dati + grafici QC + data3d.mp4 (più veloce, minimo spazio)
python scripts/athena_analysis.py data/<registrazione> --outputs minimal

# rifare solo export + grafici su dati già calcolati
python scripts/athena_analysis.py data/<registrazione> --skip-3d
```

La cartella può essere la registrazione (`data/<registrazione>`) o direttamente la main folder (`.../athena`).
La calibrazione deve essere già stata fatta (passo 3), altrimenti viene calcolata in automatico (con `--square-mm`, default 100 mm).

### Modalità batch: tutte le registrazioni di `data/`

Se la cartella passata **contiene** più registrazioni (es. `data/`), lo script le processa tutte in sequenza,
dalla calibrazione ai grafici:

```bash
# tutto su tutte le registrazioni; calibra quelle che non hanno ancora la calibrazione
python scripts/athena_analysis.py data

# solo dati + QC + data3d.mp4, più veloce
python scripts/athena_analysis.py data --outputs minimal

# ricalcolare tutto da capo (calibrazione compresa)
python scripts/athena_analysis.py data --recalibrate --redo

# scacchiera diversa da quella standard (quadrati 100 mm)
python scripts/athena_analysis.py data --square-mm 80

# scegliere le registrazioni
python scripts/athena_analysis.py data --list                                  # elenco con stato (done / to do)
python scripts/athena_analysis.py data --select roberto                        # solo quelle con "roberto" nel nome
python scripts/athena_analysis.py data --select ninepeg --select coin --exclude louis
python scripts/athena_analysis.py data --select "2026-11-10_11*" --list        # wildcard: controllare prima la selezione
```

`--select` e `--exclude` valgono anche con `--cleanup`. Un testo senza `*`/`?` viene cercato nel nome
(senza distinguere maiuscole/minuscole); con più `--select` basta che una corrisponda.

Per ogni registrazione:
1. **calibrazione** – se manca `athena/calibration/` (o con `--recalibrate`) esegue `athena_calibration.py` con
   `--square-mm` (default 100) e gli eventuali `--calib-args "..."`;
2. link ai video, ATHENA 2D/3D, export, QC, log (come per una singola registrazione).

Comportamento:
* una registrazione è riconosciuta se contiene `Take_*`, `calibration_frames/` o `athena/calibration/`;
* **le registrazioni già completate vengono saltate** (quelle con `athena/analysis/<trial>/run_info.json` di una
  run completa riuscita, con frazione di frame ≥ quella richiesta): si possono aggiungere nuove registrazioni in
  `data/` e rilanciare lo stesso comando. `--redo` forza la rielaborazione;
* se una registrazione fallisce (calibrazione impossibile, video mancanti, ...) l'errore viene scritto nel log e
  si passa alla successiva; alla fine viene stampato un riepilogo (`ok` / `skipped` / `error` e durata);
* non lanciare il batch mentre una registrazione è ancora in copia: verrebbe processata incompleta
  (di solito si ferma con un errore sul numero di video).

### Opzioni

| Opzione | Default | Significato |
|---|---|---|
| `--trial NOME` | tutti | trial da processare (ripetibile) |
| `--fraction` | `1.0` | frazione di frame (dall'inizio) da processare |
| `--processes` | `min(CPU, 9)` | processi paralleli per il 2D (uno per camera) |
| `--no-gpu` | off | MediaPipe su CPU |
| `--hand-conf` | `0.5` | confidenza minima mani MediaPipe (la GUI usa 0.9: con le mani piccole perde molte detection) |
| `--pose-conf` | `0.5` | confidenza minima posa |
| `--lfc` | `12` | cutoff [Hz] del filtro Savitzky-Golay sul 3D. A 40 fps il massimo sensato è < 20 Hz; più basso = più liscio ma picchi del tapping attenuati |
| `--outputs` | `standard` | profilo di output: `minimal`, `standard`, `full` (vedi *Output e spazio su disco*) |
| `--keep-frames` | off | conserva le immagini per frame (`images/`, `imagesrefined/`) con qualsiasi profilo |
| `--video-link` | `symlink` | come `athena/videos` punta ai `Take_*.mkv`: `symlink`, `hardlink`, `copy` |
| `--cleanup` | off | non analizza: riduce le registrazioni già elaborate al profilo `--outputs` |
| `--dry-run` | off | con `--cleanup`: mostra cosa verrebbe cancellato, senza cancellare |
| `--light` | off | vecchia opzione, equivale a `--outputs minimal` |
| `--skip-2d` | off | riusa i landmark 2D esistenti, rifà triangolazione + QC |
| `--skip-3d` | off | riusa i landmark 3D esistenti, rifà solo export + QC |
| `--run-log FILE` | `athena_runs_log.csv` | CSV dove viene aggiunta una riga per ogni run |
| `--square-mm` | `100` | lato dei quadrati della scacchiera [mm] usato per calibrare le registrazioni senza calibrazione |
| `--recalibrate` | off | ricalcola la calibrazione anche se esiste |
| `--calib-args "..."` | – | argomenti extra per `athena_calibration.py`, es. `"--origin center"` |
| `--redo` | off | batch: rielabora anche le registrazioni già completate |
| `--skip-done` | off | registrazione singola: salta i trial già completati |
| `--select PATTERN` | tutte | batch: solo le registrazioni il cui nome contiene il testo, o corrisponde alla wildcard (`"*roberto*"`); ripetibile |
| `--exclude PATTERN` | – | batch: esclude le registrazioni corrispondenti; ripetibile |
| `--list` | off | batch: elenca le registrazioni (dopo `--select`/`--exclude`) con stato, calibrazione, dimensione, durata, ed esce |

`run_info.json` contiene l'ultima run **completa** (2D+3D+QC); le run parziali (`--skip-2d`, `--skip-3d`)
scrivono `run_info_<passi>.json` e non cambiano lo stato "completato".

### Output dati

| File | Contenuto |
|---|---|
| `landmarks/<trial>/<trial>_3Dlandmarks.npy` | **risultato principale**, `(frame, 75, 3)` in mm, già filtrato |
| `landmarks/<trial>/<trial>_2Dlandmarksrefined.npy` | 3D riproiettato in ogni camera, `(cam, frame, 75, 2)` px |
| `landmarks/<trial>/camN/2Dlandmarks_{right,left,body}.npy` | detection MediaPipe grezze, `(frame, 21|33, 5)`: x, y [px], z, visibility, presence; `-1` = non rilevato |
| `landmarks/<trial>/camN/2Dworldlandmarks_*.npy` | coordinate "world" MediaPipe per camera |
| `landmarks/<trial>/camN/handedness_score.npy` | confidenza lateralità `(frame, 2)` [sinistra, destra] |
| `analysis/<trial>/<trial>_3Dlandmarks.csv` | il 3D in CSV: `frame, time_s, <landmark>_x/y/z` (celle vuote = mancante) |
| `analysis/<trial>/<trial>_hands3d.npy` | solo mani, `(frame, 2 [destra, sinistra], 21, 3)` |
| `analysis/<trial>/<trial>_3Dquality.npz` | qualità di ogni punto `(frame, 75)`: `status` 0 = mancante, 1 = interpolato, 2 = triangolato; `ncams` camere che lo vedono; `raw3d` triangolazione grezza (prima dello smoothing); `reproj_max_px` / `reproj_mean_px` errore di riproiezione sulle camere; `dev_mm` \|grezzo − finale\|; `near_gap` entro mezza finestra di smoothing da un buco; `smoothing_window` [frame]. Vedi *Qualità del dato* più sotto |
| `analysis/<trial>/<trial>_joint_angles.csv` | angoli articolari di entrambe le mani [°] per frame: `frame, time_s, R index MCP flexion, …` (50 angoli, vedi *Angoli articolari*) |

**Indici dei 75 landmark**: 0–32 corpo (MediaPipe Pose), 33–53 mano destra, 54–74 mano sinistra.
Dentro ogni mano: 0 polso, 1–4 pollice (4 = punta), 5–8 indice (8 = punta), 9–12 medio, 13–16 anulare,
17–20 mignolo. Esempio: punta del pollice destro = 33+4 = 37, punta dell'indice sinistro = 54+8 = 62.

```python
import numpy as np
d3 = np.load('athena/landmarks/<trial>/<trial>_3Dlandmarks.npy')
right, left = d3[:, 33:54], d3[:, 54:75]
tapping = np.linalg.norm(right[:, 4] - right[:, 8], axis=1)   # distanza pollice-indice [mm]
```

### Qualità del dato: cosa fa ATHENA ai dati e quanto è affidabile ogni punto

**Come ATHENA elabora i dati** (`athena.triangulaterefine`, con le opzioni di questa pipeline):
1. MediaPipe 2D su ogni camera; una detection mancata diventa NaN. Le etichette destra/sinistra vengono rese
   coerenti tra le camere.
2. **Triangolazione**: lineare (DLT) con tutte le camere che vedono il punto, se sono almeno 2. **Non c'è
   esclusione degli outlier**, né soglia sull'errore di riproiezione, né peso per la confidenza: una detection
   sbagliata in una camera sposta il punto 3D, che conta comunque come "triangolato".
3. **Smoothing**: filtro Savitzky-Golay (polinomio di grado 3) su ogni coordinata di ogni landmark. La finestra
   è `int(fps / lfc × 2 + 1)` arrotondata a dispari: **7 frame (175 ms)** a 40 fps con `--lfc 12`.
   - Prima del filtro **tutti** i buchi vengono riempiti per interpolazione lineare (valore costante all'inizio e
     alla fine del trial); dopo il filtro il NaN viene rimesso solo per i buchi di **6 o più frame**. I buchi fino
     a 5 frame (125 ms) restano riempiti.
   - Un landmark con meno frame validi della finestra non viene filtrato.
   - Gli altri affinamenti di ATHENA (passa-basso sul centro della mano, vincolo sulla lunghezza delle ossa) in
     questo passo sono **disattivati**.

Quindi il 3D finale può contenere: valori riempiti per interpolazione lineare (velocità costante e
accelerazione nulla dentro il buco), valori vicino a un buco distorti dal riempimento, e picchi da detection
sbagliate attenuati ma non eliminati.

**Affidabilità di ogni punto.** Lo script ripete la triangolazione senza smoothing e salva, per ogni frame e
landmark, le metriche in `analysis/<trial>/<trial>_3Dquality.npz`. Ogni punto riceve una classe:

| classe | significato |
|---|---|
| **reliable** (affidabile) | triangolato (≥ 2 camere) e supera tutti i controlli qui sotto |
| **suspect** (sospetto) | triangolato, ma non supera almeno un controllo |
| **interpolated** (interpolato) | visto da < 2 camere, valore riempito dallo smoothing (buco fino a 5 frame) |
| **missing** (mancante) | nessun valore nel 3D finale (buco di 6 o più frame) |

| controllo | fallisce quando (soglia di default) | cosa rivela |
|---|---|---|
| `few_cams` | il punto è visto da meno di **3** camere | nessuna ridondanza: con 2 camere una detection sbagliata non si può riconoscere |
| `reproj` | errore di riproiezione massimo sulle camere > **20 px** | una camera la cui detection 2D non è coerente con il punto 3D (detection sbagliata / mano sbagliata) |
| `outlier` | \|grezzo − finale\| > **10 mm** | picco nella triangolazione grezza appiattito dallo smoothing |
| `near_gap` | entro mezza finestra di smoothing (3 frame) da un buco | valore in parte distorto dal riempimento lineare del buco |

I default (`QUALITY_DEFAULTS` in `athena_analysis.py`) vengono dalle distribuzioni sulle prime registrazioni:
20 px e 10 mm sono circa il 95°–99° percentile. Il QC li usa per `qc_summary.json` e per il riepilogo stampato
alla fine di ogni run (`reliable … suspect … (<3 cams …, reproj …, outlier …, near gap …)`); nel notebook si
possono cambiare in modo interattivo.

```python
import numpy as np, sys
sys.path.insert(0, 'scripts'); import athena_analysis as AA
q = dict(np.load('data/<registrazione>/athena/analysis/<trial>/<trial>_3Dquality.npz'))
cls, reasons = AA.quality_classes(q, min_cams=3, max_reproj_px=20, max_dev_mm=10)   # (frame, 75)
affidabili = cls == 3                     # 0 mancante, 1 interpolato, 2 sospetto, 3 affidabile
reproj_alta = reasons['reproj']           # anche 'few_cams', 'outlier', 'near_gap'
```

Per le registrazioni elaborate prima di queste metriche: `python scripts/athena_analysis.py data --skip-3d --redo`
(rifà solo export e QC, pochi secondi ciascuna). In alternativa le calcola il notebook al primo caricamento.

### Angoli articolari – `scripts/hand_angles.py`

Calcolati dal 3D finale per entrambe le mani, esportati in `analysis/<trial>/<trial>_joint_angles.csv` e
disponibili nel notebook (grandezza *joint angles*). In gradi; i nomi iniziano con `R` / `L`.

| angolo | definizione | segno |
|---|---|---|
| **MCP flexion** (indice, medio, anulare, mignolo) | metacarpo (polso → MCP) vs falange prossimale (MCP → PIP), attorno all'asse di flessione della mano (MCP indice → MCP mignolo) | 0 = dritto, + flessione, − iperestensione |
| **PIP flexion** | falange prossimale vs falange media, stesso asse | idem |
| **DIP flexion** | falange media vs falange distale, stesso asse | idem |
| **TAM** (total active motion) | MCP + PIP + DIP di un dito | |
| **spread** indice–medio, medio–anulare, anulare–mignolo | angolo tra le falangi prossimali di dita adiacenti, sul piano del palmo (abduzione MCP) | ≥ 0 |
| **thumb MCP / IP flexion** | segmenti consecutivi del pollice (CMC → MCP → IP → punta) | 0 = dritto, ≥ 0 |
| **thumb abduction** | 1° (pollice) vs 2° (indice) metacarpo | ≥ 0 |
| **wrist flexion/extension** | avambraccio (gomito → polso) vs mano (polso → MCP medio), attorno all'asse di flessione | + flessione palmare, − estensione |
| **wrist radial/ulnar deviation** | stessi vettori sul piano del palmo | + radiale (lato pollice), − ulnare |
| **forearm rotation** | normale del palmo vs verticale, attorno all'asse dell'avambraccio (Z mondo verso l'alto) | 0 palmo in giù … 90 verticale … 180 palmo in su |

Perché questi: la flessione MCP e PIP sono le misure principali della funzione delle dita (presa, tapping); la DIP
completa la catena del dito e il TAM la riassume (misura clinica standard); lo spread descrive l'apertura della
mano; abduzione e flessioni del pollice descrivono presa e opposizione (le distanze pollice–dita sono nel notebook
come *distance A–B*); gli angoli di polso e avambraccio descrivono la postura in cui lavora la mano.

Note:
* il segno della flessione è lo stesso per le due mani (l'asse di flessione è specchiato per la sinistra);
  verificato sulle registrazioni: PIP e MCP quasi sempre positivi, DIP raramente un po' negativo (iperestensione
  fisiologica);
* il landmark 1 del pollice di MediaPipe (CMC) approssima l'articolazione trapezio-metacarpale: la
  flessione/opposizione della CMC del pollice non viene calcolata come angolo;
* **gli angoli di polso e avambraccio usano il gomito di MediaPipe Pose**: sulle registrazioni attuali il suo
  errore di riproiezione è grande (mediana 30–450 px, contro 5–9 px della mano), quindi risultano *suspect* e vanno
  considerati indicativi;
* nel notebook la qualità di un angolo in ogni frame è la classe peggiore tra i landmark che usa.

```python
import sys, numpy as np; sys.path.insert(0, 'scripts')
from hand_angles import hand_angles
angles, used = hand_angles(np.load('.../<trial>_3Dlandmarks.npy'), 'R')   # {nome: (frame,) °}, {nome: [landmark]}
```

### Grafici di qualità (`analysis/<trial>/`)

| File | Cosa guardare |
|---|---|
| `01_acquisition.png` | intervalli tra frame per camera (devono essere 25 ms a 40 fps), frame persi, skew tra camere. Dai timestamp PTP di `Metadata_*.json` |
| `02_detection_2d.png` | per camera e nel tempo: dove MediaPipe ha rilevato mano destra, sinistra, corpo (% in etichetta) |
| `03_coverage.png` | numero di camere che vedono ogni mano; rosso = mancante nel 3D finale; blu = buco **riempito dallo smoothing** (il 3D grezzo mancava) |
| `04_gaps.png` + `gaps.csv` | timeline e istogramma della durata dei buchi, nel 3D grezzo e finale; il CSV ha inizio/fine/durata di ogni buco |
| `05_timeseries_<mano>.png` | x, y, z di polso / punta pollice / punta indice (linea = finale, punti = triangolazione grezza); velocità con spike (× = oltre mediana + 6·MAD); distanza pollice–indice; camere per punto |
| `06_reprojection.png` | errore di riproiezione [px] per camera e nel tempo: camere con errore alto = calibrazione o detection sbagliate |
| `07_bone_lengths.png` | lunghezza delle ossa nel tempo: dovrebbe essere costante; variabilità alta = rumore o scambi di mano |
| `qc_summary.json` | tutte le metriche numeriche (detection %, buchi, errori, spike, CV ossa) e, per mano, % affidabili / sospetti / interpolati / mancanti con la quota di ogni controllo fallito (`quality_thresholds` = soglie usate) |
| `run_options.json`, `run_info.json` | opzioni e info/tempi di questa run |

Note di interpretazione:
* *raw 3D* = triangolazione diretta delle detection (servono ≥ 2 camere). *final 3D* = output ATHENA dopo
  lo smoothing, che **interpola i buchi fino a 5 frame**: un dato "valido" nel finale può essere interpolato.
* Gli spike di velocità durante il tapping possono essere movimenti reali: valutare insieme alla traccia
  della distanza pollice–indice.

### Output visivi ATHENA

* `videos_processed/<trial>/data3d.mp4` – scheletro 3D (**sempre** prodotto)
* `videos_processed/<trial>/camN.mp4` – detection MediaPipe 2D (profili `standard`, `full`)
* `videos_processed/<trial>/camN_refined.mp4` – 3D riproiettato su ogni camera (profili `standard`, `full`)
* immagini per frame da cui sono fatti i video: `images/<trial>/camN/*.png`, `imagesrefined/<trial>/{camN,data3d}/*.jpg`
  (solo profilo `full` o `--keep-frames`; altrimenti cancellate dopo aver creato i video)
* video affiancato di tutte le camere: `athena-montage` (o `python -m athena.montage`, richiede le immagini)

### Output e spazio su disco

| Profilo `--outputs` | Cosa resta oltre a landmark, analysis/QC e `data3d.mp4` | `athena/` per 46 s, 9 camere |
|---|---|---|
| `minimal` | niente (nessun video di overlay) | ~0,13 GB |
| `standard` (default) | video di overlay `camN.mp4` e `camN_refined.mp4` | ~0,3 GB |
| `full` | come `standard` + tutte le immagini per frame | ~4,7 GB |

Prima di queste modifiche la stessa registrazione occupava ~9,3 GB in `athena/` (copia dei video + immagini).

Con tutti i profili i `Take_*.mkv` originali non vengono toccati e `athena/videos` contiene solo link.
`minimal` è anche più veloce (niente overlay 2D e riproiettati: risparmia la scrittura di ~1 immagine per
camera per frame); lo scheletro 3D (`data3d.mp4`) resta la parte più lenta del passo 3D.

Se una mano ha un landmark **mai** triangolato in tutto il trial, ATHENA non può disegnare lo scheletro 3D:
`data3d.mp4` viene saltato con un warning (`data3d_video = False` nel log).

**Pulizia di registrazioni già elaborate** (senza rifare l'analisi):

```bash
python scripts/athena_analysis.py data --cleanup --dry-run          # mostra cosa toglierebbe e quanto
python scripts/athena_analysis.py data --cleanup                    # applica il profilo standard
python scripts/athena_analysis.py data --cleanup --outputs minimal  # tiene solo dati, QC e data3d.mp4
```

`--cleanup` (su una registrazione o su tutta `data/`):
* sostituisce le copie mp4 in `athena/videos/` con link ai `Take_*.mkv`, dopo aver verificato che abbiano lo
  stesso numero di frame;
* cancella le immagini per frame (se un video previsto dal profilo manca, prima lo crea dalle immagini);
* con `minimal` cancella anche i video di overlay `camN*.mp4`; `data3d.mp4` non viene mai cancellato
  (se manca, il messaggio suggerisce di rilanciare con `--skip-2d`).

Differenza tra `--cleanup` e `--cleanup --outputs minimal`: entrambi trasformano le copie video in link e
cancellano le immagini per frame (~9 GB per registrazione); cambia solo `videos_processed/<trial>/`:

| | `--cleanup` (standard) | `--cleanup --outputs minimal` |
|---|---|---|
| `data3d.mp4` – scheletro 3D | tenuto | tenuto |
| `camN.mp4` – video originale con i punti trovati da MediaPipe | tenuto | cancellato |
| `camN_refined.mp4` – video con il 3D finale riproiettato | tenuto | cancellato |

I 18 video di overlay occupano ~150–170 MB per registrazione e servono a controllare a occhio la qualità del
tracking in ogni camera: conviene tenerli (`--cleanup` semplice). Se cancellati si rigenerano rilanciando
l'analisi (`--skip-2d` rifà solo i `_refined`; i `camN.mp4` richiedono anche il 2D).

### Dati su un PC remoto (SSH)

Se le registrazioni sono su un altro PC, senza GPU adatta, l'elaborazione si fa su questo PC lavorando via SSH,
senza tenere una copia permanente dei dati. Banda necessaria: ~20 MB/s medi, perché ogni mkv viene letto circa
due volte, la prima nel 2D e la seconda per gli overlay `_refined`.

**A. Cartella remota montata con SSHFS** (consigliato in LAN gigabit). Non si copia nulla: lo script legge gli
mkv via rete e scrive i risultati direttamente in `athena/` sul PC remoto.

```bash
sudo apt install sshfs                                   # una volta sola
mkdir -p ~/remote_data
sshfs utente@pc-remoto:/percorso/data ~/remote_data -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

conda activate athena
python scripts/athena_analysis.py ~/remote_data --list              # cosa c'è sul remoto e cosa è da fare
python scripts/athena_analysis.py ~/remote_data                     # tutte quelle ancora da fare
python scripts/athena_analysis.py ~/remote_data --select roberto --select 2026-11-12   # solo alcune

fusermount -u ~/remote_data                              # smonta a fine lavoro
```

Se la connessione cade, la registrazione in corso va in errore e il batch passa alla successiva; rilanciando lo
stesso comando, quelle già completate vengono saltate.

**B. Copia temporanea, una registrazione alla volta** (rete lenta o instabile, VPN, internet). Ogni
registrazione viene copiata in locale (~5 GB), elaborata e cancellata; sul PC remoto torna solo `athena/`.

```bash
HOST=utente@pc-remoto
RDIR=/percorso/data
TMP=~/athena_tmp && mkdir -p $TMP
conda activate athena
SEL='*'                       # registrazioni da elaborare: wildcard, es. '*roberto*' o '2026-11-12_*'
ssh $HOST "cd $RDIR && ls -d $SEL"          # controllare prima la selezione
for rec in $(ssh $HOST "cd $RDIR && ls -d $SEL"); do
    # salta le registrazioni già completate sul remoto
    ssh $HOST "ls $RDIR/$rec/athena/analysis/*/run_info.json" >/dev/null 2>&1 && continue
    rsync -a --info=progress2 --exclude 'athena/images' --exclude 'athena/imagesrefined' \
          "$HOST:$RDIR/$rec" "$TMP/" || continue
    python scripts/athena_analysis.py "$TMP/$rec"
    rsync -a "$TMP/$rec/athena/" "$HOST:$RDIR/$rec/athena/"      # risultati -> PC remoto
    rm -rf "$TMP/$rec"                                            # nessuna copia permanente
done
```

Note:
* per un elenco preciso invece della wildcard: `for rec in 2026-11-10_111114_roberto_ninepeg_01 2026-11-10_111503_roberto_coin_01; do`;
* il log dei tempi `athena_runs_log.csv` resta su questo PC; con SSHFS i tempi includono anche la lettura via rete;
* i link `athena/videos/<trial>/cam_N.mp4` sono relativi, quindi restano validi anche sul PC remoto;
* per accedere senza password: `ssh-keygen` e poi `ssh-copy-id utente@pc-remoto`.

### Raccolta dei video 3D – `collect_data3d.py`

Copia tutti i `videos_processed/<trial>/data3d.mp4` in un'unica cartella `videos3ddata/` (nella cartella del
progetto), rinominati con il nome della registrazione: `videos3ddata/<registrazione>.mp4`
(`<registrazione>__<trial>.mp4` se una registrazione ha più trial).

```bash
python scripts/collect_data3d.py                       # cerca in data/
python scripts/collect_data3d.py ~/remote_data         # altra cartella, es. il mount SSHFS
python scripts/collect_data3d.py --select roberto      # solo alcune registrazioni
python scripts/collect_data3d.py --mode symlink        # link invece di copie (0 byte, serve la cartella dati)
python scripts/collect_data3d.py --out /percorso/cartella
```

I video già raccolti e invariati vengono saltati: si può rilanciare dopo aver elaborato nuove registrazioni.

---

## 5. Log dei tempi e stima della durata – `athena_runs_log.csv`

Ogni esecuzione di `athena_analysis.py` aggiunge una riga (anche se fallisce: `status` = `error`/`interrupted`).
Una copia della riga è in `analysis/<trial>/run_info.json`.

| Gruppo | Colonne |
|---|---|
| run | `run_start`, `status`, `error`, `recording`, `trial`, `steps` (2d/3d/qc eseguiti) |
| video | `n_cameras`, `width`, `height`, `fps`, `codec`, `bitrate_mbps_per_cam`, `videos_total_mb`, `video_frames`, `video_duration_s` |
| opzioni | `fraction`, `processed_frames`, `processed_duration_s`, `processed_camera_frames`, `use_gpu`, `processes`, `hand_conf`, `pose_conf`, `lfc_hz`, `save_*_images_video`, `outputs_profile`, `keep_frames`, `video_link` |
| tempi [s] | `t_prepare_s`, `t_2d_s`, `t_3d_s` (include immagini/video), `t_qc_export_s`, `t_total_s` |
| throughput | `s_per_camera_frame_2d`, `fps_per_camera_2d`, `s_per_frame_3d`, `realtime_factor_total` (= tempo di calcolo / durata processata) |
| dati estratti | `right/left_hand_valid_pct`, `right/left_hand_gaps`, `n_images_written` (immagini generate, anche se poi cancellate), `n_videos_written`, `data3d_video`, `output_*_mb` |
| macchina | `hostname`, `cpu_model`, `cpu_count`, `ram_gb`, `gpu`, `mediapipe_version`, `athena_version` |

**Stima a priori** per un nuovo video (stessa macchina e stesse opzioni):

```
t_2d ≈ s_per_camera_frame_2d × n_frame × n_camere
t_3d ≈ s_per_frame_3d × n_frame                      (dominato dalle immagini/video se attivi)
t_totale ≈ realtime_factor_total × durata_video_s
spazio disco ≈ output_images_videos_mb / processed_frames × n_frame
```

```python
import pandas as pd
log = pd.read_csv('athena_runs_log.csv')
ok = log[(log.status == 'ok') & (log.steps == '2d+3d+qc')]
print(ok.groupby(['outputs_profile', 'use_gpu'])[['s_per_camera_frame_2d', 's_per_frame_3d',
                                                   'realtime_factor_total']].median())
```

---

## 5b. Visualizzazione interattiva – `notebooks/landmarks3d_viewer.ipynb`

Notebook Jupyter per esplorare i tracciati 3D e la loro qualità. Il notebook (testi, controlli, grafici ed
export) è in inglese; qui sotto tra parentesi i nomi che compaiono nell'interfaccia. Si sceglie con i controlli:
* **registrazione**;
* **landmark**, filtrati per gruppo (*Group*: right hand, left hand, body, all);
* **grandezza** (*Quantity*):
  - posizione x/y/z;
  - velocità (modulo o per componenti) e accelerazione;
  - distanza A–B e angolo A–B–C, per esempio la flessione di un dito;
  - **angoli articolari** (*joint angles*: flessione MCP, PIP, DIP, TAM, spread, pollice, polso, avambraccio),
    scelti nella lista *Joint angles* che sostituisce quella dei landmark; la scheda *Statistics* mostra allora, per
    ogni angolo, qualità e ampiezza di movimento (ROM) su tutti i frame e sui soli frame affidabili;
  - **lunghezza dell'osso** (*bone length*) che termina in ogni landmark selezionato (es. `R_index_tip` = falange
    distale): dovrebbe restare costante;
  - qualità: numero di camere, **errore di riproiezione** (massimo / medio sulle camere), **deviazione grezzo–finale**,
    con la soglia disegnata come linea tratteggiata;
* **intervallo di tempo**;
* **soglie di qualità** (pannello *Quality thresholds*): camere minime, errore di riproiezione massimo, deviazione
  massima, frame vicino a un buco. Tutti i grafici e le tabelle si aggiornano.

Le quattro schede:

| Scheda | Contenuto |
|---|---|
| Time series | grafici zoomabili; bande **viola = sospetto**, **arancioni = interpolato**, **rosse = mancante**; passando col mouse si vedono classe e controlli falliti; opzionale la triangolazione grezza sovrapposta (posizione, lunghezza ossa) |
| 3D trajectory | percorsi nel tempo + scheletro delle mani a un frame scelto, giunti colorati per classe (col mouse: camere, errore di riproiezione, deviazione) |
| Quality | mappa nel tempo, per landmark, di classe, numero di camere, errore di riproiezione o deviazione grezzo–finale |
| Statistics | per landmark: % affidabile/sospetto/interpolato/mancante, % di ogni controllo fallito, camere medie, errore di riproiezione mediano, p99 della deviazione, buchi, ampiezza del movimento, velocità |

*Save plot* scrive l'HTML interattivo e *Save data* scrive il CSV con, per ogni landmark, la classe
(`status_<landmark>`) e i controlli falliti (`flags_<landmark>`, es. `few_cams;reproj`) di ogni punto, entrambi in
`exports/`. Le ultime celle del notebook mostrano una tabella di affidabilità di tutte le registrazioni.

```bash
~/miniforge3/envs/athena/bin/jupyter lab notebooks/landmarks3d_viewer.ipynb
```

In VSCode: aprire il notebook → *Select Kernel* → *Jupyter Kernel* → **Python (athena)**.

Tutto ciò che si può voler cambiare (cartella dati, registrazione, landmark e grandezza iniziali, soglie di
affidabilità, media mobile, altezza grafici, cartella di export) è nella cella **Parameters**. Le stesse funzioni (`load_trial`,
`timeseries_figure`, `compute_series`, `stats_table`, ...) si possono chiamare anche da codice: vedi l'ultima
sezione del notebook.

---

## 6. GUI originale di ATHENA (opzionale)

```bash
python scripts/athena_gui.py        # GUI con font ingranditi x2 (schermi HiDPI / Wayland)
python scripts/athena_gui.py 1.5    # altro fattore
```

Nella GUI selezionare `data/<registrazione>/athena` come main folder. Attenzione: i default della GUI
(confidenza 0.9) perdono molte detection delle mani in queste riprese, e lo scheletro 3D per frame
blocca a lungo l'interfaccia. Per l'uso normale preferire `athena_analysis.py`.

---

## 7. Problemi noti

| Problema | Causa / soluzione |
|---|---|
| `cannot import name 'solutions' from 'mediapipe'` | `pip install "mediapipe==0.10.21"` |
| `No module named 'moviepy.editor'` | `pip install "moviepy<2"` |
| VSCode: la cella dei controlli del notebook resta "in esecuzione" e i controlli non compaiono (log Jupyter: *Widget Error: Failed to access CDN*) | VSCode non è autorizzato a scaricare il codice dei widget: in `.vscode/settings.json` del progetto c'è già `"jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"]`; ricaricare la finestra (*Developer: Reload Window*) e riavviare il kernel. In alternativa usare JupyterLab |
| VSCode: `Import "numpy" could not be resolved`, o `python` che non trova i pacchetti | VSCode usa la vecchia `.venv` (Python 3.14, vuota) invece dell'ambiente conda: cancellare `.venv` e scegliere l'interprete `~/miniforge3/envs/athena/bin/python` (*Python: Select Interpreter*) |
| `conda: command not found` dopo l'installazione | `~/miniforge3/bin/conda init bash && source ~/.bashrc` |
| scritte minuscole nella GUI | usare `scripts/athena_gui.py` |
| mani rilevate in poche camere | le camere inquadrano la stanza: le mani sono piccole (60–100 px). Abbassare `--hand-conf`; controllare `02_detection_2d.png` |
| scala 3D sbagliata | `--square-mm` errato in calibrazione: rifare la calibrazione e poi `athena_analysis.py` (anche `--skip-2d`: il 2D non dipende dalla scala) |
| "ERROR: no calibration" | eseguire prima `athena_calibration.py` |
| numero di video ≠ numero di calibrazioni | ATHENA richiede un file di calibrazione per ogni video del trial |
