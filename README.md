# DEXTER kinematics – ATHENA pipeline for 3D hand tracking

*Italian version: [README_IT.md](README_IT.md)*

Pipeline to reconstruct hands (and body) in 3D from multi-camera recordings (9 Basler cameras, `multicam_capture`)
using [ATHENA](https://github.com/neural-control-and-computation-lab/athena) (MediaPipe + triangulation),
with checkerboard calibration and quality analysis of the extracted data.

```
scripts/
├── athena_calibration.py   # 1. multi-camera calibration from the checkerboard -> ATHENA format
├── athena_analysis.py      # 2. headless ATHENA pipeline + quality plots + run-time log (single / batch)
├── collect_data3d.py       #    collects all data3d.mp4 videos into videos3ddata/
├── hand_angles.py          #    joint angles (MCP, PIP, DIP, TAM, spread, thumb, wrist, forearm)
├── athena_gui.py           #    (optional) original ATHENA GUI with enlarged fonts
└── requirements.txt
notebooks/
└── landmarks3d_viewer.ipynb  # 3. interactive visualization of the 3D tracks and their quality
data/                       # recordings (one folder per recording)
videos3ddata/               # 3D skeleton videos, one per recording (collect_data3d.py)
exports/                    # plots (HTML) and data (CSV) saved from the notebook
diario/                     # work diary
athena_runs_log.csv         # log of all runs (created automatically)
.vscode/settings.json       # VSCode: conda interpreter + widget download for the notebook
```

---

## 1. Installation (once)

ATHENA requires **Python 3.12**. A conda environment is used.

```bash
# Miniforge (if conda is not installed)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash && source ~/.bashrc

# Environment
conda create -n athena python=3.12
conda activate athena
pip install athena-tracking opencv-python
pip install jupyterlab ipywidgets plotly pandas ipykernel "numpy==1.26.4"   # visualization notebook
python -m ipykernel install --user --name athena --display-name "Python (athena)"
pip install "mediapipe==0.10.21" "moviepy<2"     # versions compatible with ATHENA 0.9 (see below)
conda install -c conda-forge ffmpeg
```

> **Why pinned versions**: `mediapipe` ≥ 0.10.30 removed `mediapipe.solutions` (error
> *cannot import name 'solutions'*), and `moviepy` 2.x removed `moviepy.editor` (used by `athena-montage`).
> pip reports numpy conflicts with `astropy`/`tifffile`/`opencv-python 5`: they can be ignored, ATHENA does not use them.

Check:
```bash
conda activate athena
python -c "import athena.labels2d, athena.triangulaterefine, athena.montage; print('ok')"
```

All the following commands must be run with the environment active (`conda activate athena`) from the project
folder `~/Desktop/DEXTER_kinematics`.

---

## 2. Data layout

A `multicam_capture` recording (`data/<recording>/`) contains:

```
data/2026-11-10_112048_louis_finertapping_01/
├── Take_<stamp>-M<serial>.mkv           # one video per camera (MJPEG, 1920x1200, 40 fps)
├── Metadata_<stamp>-M<serial>.json      # PTP timestamp of every frame (used for acquisition QC)
├── metadata.json, recording_verification.json, ...
└── calibration_frames/<capture>/
    ├── cam_<N>_frame<K>.png             # synchronized checkerboard screenshots
    └── camera_config.json               # camera name <-> serial, focal length
```

The scripts create the ATHENA *main folder* inside the recording:

```
data/<recording>/athena/
├── calibration/cam_1.yaml ... cam_9.yaml      # from athena_calibration.py
├── calibration_check/cam_N.jpg                # visual calibration check
├── calibration_report.json                    # errors, camera <-> serial <-> video mapping
├── videos/<trial>/cam_1.mp4 ... cam_9.mp4     # links to Take_*.mkv (0 bytes), names/order for ATHENA
├── landmarks/<trial>/                         # ATHENA data (per-camera 2D, 3D, reprojected 2D)
├── videos_processed/<trial>/data3d.mp4        # 3D skeleton (always)
├── videos_processed/<trial>/camN[_refined].mp4  # 2D overlay / reprojected 3D (standard and full profiles)
├── images/<trial>/, imagesrefined/<trial>/    # per-frame images: only with --outputs full / --keep-frames
└── analysis/<trial>/                          # export + quality plots + run_info.json
```

`<trial>` is the recording name without date/time (e.g. `louis_finertapping_01`), or the name of the
subfolders already present in `athena/videos/`.

**Recording recommendations** (from the first recordings):
* start recording **a couple of seconds before** the movement begins and stop **a couple of seconds after** it
  ends: the first and last frames often have the hands partly out of view or not yet detected, and the
  smoothing is less reliable at the edges of the trial;
* avoid complete occlusions of a hand (body, other hand, objects) as far as the task allows: an occluded hand
  becomes *missing* (gaps of 6+ frames are not filled);
* hands should be seen by **at least 3 cameras** for most of the time (check `qc_summary.json` or the notebook:
  with 2 cameras a wrong detection cannot be recognized).

---

## 3. Calibration – `athena_calibration.py`

Computes camera intrinsics and extrinsics from the checkerboard screenshots and writes the files in the format
read by ATHENA.

```bash
python scripts/athena_calibration.py data/<recording>                    # lab board, 100 mm squares
python scripts/athena_calibration.py data/<recording> --square-mm 80     # different board
```

**`--square-mm`** (default **100**, the lab board has 10 cm squares): the *real* side of one checkerboard square,
in mm. Change it only when using a different board.
It sets the scale of all 3D measurements: a wrong value scales all distances proportionally.

| Option | Default | Meaning |
|---|---|---|
| `--inner-corners` | `4x6` | inner corners (squares − 1 per side); 4x6 = 5x7 board |
| `--format` | `yaml` | `yaml` (JARVIS, one file per camera) or `toml` (Anipose) |
| `--out` | `<recording>/athena` | output main folder |
| `--origin` | `corner` | world origin: board corner next to the red edge, or `center` |
| `--intrinsics-mode` | `focal` | `focal`: fit focal length only; `focal_k1`: + radial distortion; `nominal`: focal from lens/pixel pitch; `full`: needs many board poses |
| `--intrinsics-dir` | – | pre-calibrated intrinsics (more accurate): only extrinsics are computed |
| `--pixel-size-um` | `5.86` | sensor pixel pitch (Basler acA1920-40gc) |
| `--max-focal-deviation` | `0.15` | if the fitted focal length deviates more than 15% from nominal, use nominal |
| `--flip-cams` | – | e.g. `3,7`: cameras whose axes are rotated by 180° in the checks (red edge misread) |
| `--ref-frame`, `--frames-dir` | auto | choice of screenshot frame/folder |
| `--skip-video-check` | off | do not compare videos with screenshots |

**Reference frame** (mm): origin on the checkerboard lying on the table, X along the long side
(parallel to the red edge), Y along the short side, **Z up**.

**Check**: open `athena/calibration_check/cam_N.jpg` (axes must be consistent across cameras)
and read the errors in `athena/calibration_report.json` (`triangulation_rms_mm`).

> With a single frame per camera the intrinsics are estimates: for best accuracy calibrate the
> intrinsics once with many board poses and pass them with `--intrinsics-dir`.

---

## 4. Tracking and analysis – `athena_analysis.py`

Runs in sequence, without GUI:

1. **prepare** – links `Take_*.mkv` as `athena/videos/<trial>/cam_N.mp4` (renamed and ordered like the
   calibration files; ATHENA pairs videos and calibrations in alphabetical order). No copy: symbolic links.
2. **ATHENA 2D** – MediaPipe (hands + pose) on all cameras in parallel (GPU).
3. **ATHENA 3D** – left/right correction, triangulation, smoothing, reprojection, images and videos.
4. **export** – CSV and npy of the 3D landmarks.
5. **QC** – quality plots and statistics.
6. **log** – run time and run info in `athena_runs_log.csv`.

### Video preparation (automatic)

ATHENA only looks for `.avi`/`.mp4` files and pairs videos and calibrations in alphabetical order, so the
`Take_<stamp>-M<serial>.mkv` files must be renamed. This is done by the first step of the script
(`prepare_videos`), on every run:

* serial → `cam_N` mapping read from `athena/calibration_report.json` (`athena_order` section, written by
  `athena_calibration.py`): this is why calibration must be done first;
* `athena/videos/<trial>/cam_N.mp4` is a **relative symbolic link** to the original `Take_*.mkv`: no extra
  space, no loss (OpenCV detects the mkv container regardless of the extension). The link is relative, so it
  stays valid when the whole recording folder is moved/copied;
* `--video-link hardlink` uses hard links (same disk), `--video-link copy` the old behaviour
  (`ffmpeg -c copy` into an mp4, doubles the video space);
* videos already present in `athena/videos/<trial>/` are not redone; if the folder is deleted (or the links
  break, e.g. copy to a FAT/exFAT disk that does not support links) it is recreated on the next run;
* checks that the number of videos equals the number of calibration files and that all have the same
  number of frames (warning otherwise).

### Usage

```bash
# quick test on 10% of the frames
python scripts/athena_analysis.py data/<recording> --fraction 0.1

# full run (standard output profile)
python scripts/athena_analysis.py data/<recording>

# no argument: a dialog opens to choose the folder
python scripts/athena_analysis.py

# data + QC plots + data3d.mp4 only (faster, minimum space)
python scripts/athena_analysis.py data/<recording> --outputs minimal

# redo only export + plots on already computed data
python scripts/athena_analysis.py data/<recording> --skip-3d
```

The folder can be the recording (`data/<recording>`) or the main folder itself (`.../athena`).
Calibration must already be done (step 3), otherwise it is computed automatically (with `--square-mm`, default 100 mm).

### Batch mode: all recordings in `data/`

If the given folder **contains** several recordings (e.g. `data/`), the script processes all of them in sequence,
from calibration to plots:

```bash
# everything on all recordings; calibrates those that are not calibrated yet
python scripts/athena_analysis.py data

# data + QC + data3d.mp4 only, faster
python scripts/athena_analysis.py data --outputs minimal

# recompute everything from scratch (calibration included)
python scripts/athena_analysis.py data --recalibrate --redo

# board different from the standard one (100 mm squares)
python scripts/athena_analysis.py data --square-mm 80

# choosing the recordings
python scripts/athena_analysis.py data --list                                  # list with status (done / to do)
python scripts/athena_analysis.py data --select roberto                        # only those with "roberto" in the name
python scripts/athena_analysis.py data --select ninepeg --select coin --exclude louis
python scripts/athena_analysis.py data --select "2026-11-10_11*" --list        # wildcard: check the selection first
```

`--select` and `--exclude` also apply to `--cleanup`. A text without `*`/`?` is searched in the name
(case-insensitive); with several `--select` one match is enough.

For each recording:
1. **calibration** – if `athena/calibration/` is missing (or with `--recalibrate`) runs `athena_calibration.py` with
   `--square-mm` (default 100) and optional `--calib-args "..."`;
2. video links, ATHENA 2D/3D, export, QC, log (as for a single recording).

Behaviour:
* a folder is recognised as a recording if it contains `Take_*`, `calibration_frames/` or `athena/calibration/`;
* **recordings already completed are skipped** (those with an `athena/analysis/<trial>/run_info.json` from a
  successful full run, with frame fraction ≥ the requested one): you can add new recordings to `data/` and run
  the same command again. `--redo` forces re-processing;
* if a recording fails (calibration impossible, missing videos, ...) the error is written to the log and the
  next recording is processed; a summary is printed at the end (`ok` / `skipped` / `error` and duration);
* do not start the batch while a recording is still being copied: it would be processed incomplete
  (usually it stops with an error on the number of videos).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--trial NAME` | all | trial to process (repeatable) |
| `--fraction` | `1.0` | fraction of frames (from the start) to process |
| `--processes` | `min(CPU, 9)` | parallel processes for 2D (one per camera) |
| `--no-gpu` | off | MediaPipe on CPU |
| `--hand-conf` | `0.5` | MediaPipe minimum hand confidence (the GUI uses 0.9: with small hands it loses many detections) |
| `--pose-conf` | `0.5` | minimum pose confidence |
| `--lfc` | `12` | cutoff [Hz] of the Savitzky-Golay filter on 3D. At 40 fps the sensible maximum is < 20 Hz; lower = smoother but tapping peaks attenuated |
| `--outputs` | `standard` | output profile: `minimal`, `standard`, `full` (see *Outputs and disk space*) |
| `--keep-frames` | off | keep the per-frame images (`images/`, `imagesrefined/`) with any profile |
| `--video-link` | `symlink` | how `athena/videos` refers to `Take_*.mkv`: `symlink`, `hardlink`, `copy` |
| `--cleanup` | off | no analysis: shrink recordings already processed to the `--outputs` profile |
| `--dry-run` | off | with `--cleanup`: show what would be deleted, without deleting |
| `--light` | off | old option, same as `--outputs minimal` |
| `--skip-2d` | off | reuse existing 2D landmarks, redo triangulation + QC |
| `--skip-3d` | off | reuse existing 3D landmarks, redo only export + QC |
| `--run-log FILE` | `athena_runs_log.csv` | CSV where one row is appended per run |
| `--square-mm` | `100` | checkerboard square side [mm] used to calibrate recordings without calibration |
| `--recalibrate` | off | recompute calibration even if present |
| `--calib-args "..."` | – | extra arguments for `athena_calibration.py`, e.g. `"--origin center"` |
| `--redo` | off | batch: re-process recordings already completed |
| `--skip-done` | off | single recording: skip trials already completed |
| `--select PATTERN` | all | batch: only recordings whose name contains the text, or matches the wildcard (`"*roberto*"`); repeatable |
| `--exclude PATTERN` | – | batch: exclude matching recordings; repeatable |
| `--list` | off | batch: list the recordings (after `--select`/`--exclude`) with status, calibration, size, duration, then exit |

`run_info.json` holds the last **full** run (2D+3D+QC); partial runs (`--skip-2d`, `--skip-3d`)
write `run_info_<steps>.json` and do not change the "completed" state.

### Data outputs

| File | Content |
|---|---|
| `landmarks/<trial>/<trial>_3Dlandmarks.npy` | **main result**, `(frame, 75, 3)` in mm, already filtered |
| `landmarks/<trial>/<trial>_2Dlandmarksrefined.npy` | 3D reprojected into each camera, `(cam, frame, 75, 2)` px |
| `landmarks/<trial>/camN/2Dlandmarks_{right,left,body}.npy` | raw MediaPipe detections, `(frame, 21|33, 5)`: x, y [px], z, visibility, presence; `-1` = not detected |
| `landmarks/<trial>/camN/2Dworldlandmarks_*.npy` | MediaPipe "world" coordinates per camera |
| `landmarks/<trial>/camN/handedness_score.npy` | handedness confidence `(frame, 2)` [left, right] |
| `analysis/<trial>/<trial>_3Dlandmarks.csv` | 3D as CSV: `frame, time_s, <landmark>_x/y/z` (empty cells = missing) |
| `analysis/<trial>/<trial>_hands3d.npy` | hands only, `(frame, 2 [right, left], 21, 3)` |
| `analysis/<trial>/<trial>_3Dquality.npz` | quality of every point `(frame, 75)`: `status` 0 = missing, 1 = interpolated, 2 = triangulated; `ncams` cameras that see it; `raw3d` raw triangulation (before smoothing); `reproj_max_px` / `reproj_mean_px` reprojection error over the cameras; `dev_mm` \|raw − final\|; `near_gap` within half a smoothing window of a gap; `smoothing_window` [frames]. See *Data quality* below |
| `analysis/<trial>/<trial>_joint_angles.csv` | joint angles of both hands [°] per frame: `frame, time_s, R index MCP flexion, …` (50 angles, see *Joint angles*) |

**Indices of the 75 landmarks**: 0–32 body (MediaPipe Pose), 33–53 right hand, 54–74 left hand.
Within each hand: 0 wrist, 1–4 thumb (4 = tip), 5–8 index (8 = tip), 9–12 middle, 13–16 ring,
17–20 pinky. Example: right thumb tip = 33+4 = 37, left index tip = 54+8 = 62.

```python
import numpy as np
d3 = np.load('athena/landmarks/<trial>/<trial>_3Dlandmarks.npy')
right, left = d3[:, 33:54], d3[:, 54:75]
tapping = np.linalg.norm(right[:, 4] - right[:, 8], axis=1)   # thumb-index distance [mm]
```

### Data quality: what ATHENA does to the data, and how reliable each point is

**How ATHENA processes the data** (`athena.triangulaterefine`, with the options of this pipeline):
1. MediaPipe 2D on every camera; a missed detection becomes NaN. Left/right hand labels are made consistent
   across cameras.
2. **Triangulation**: linear (DLT) with all cameras that see the point, if there are at least 2. There is **no
   outlier rejection**, no reprojection-error threshold and no confidence weighting: a wrong detection in one
   camera moves the 3D point and the point still counts as "triangulated".
3. **Smoothing**: Savitzky-Golay filter (polynomial order 3) on each coordinate of each landmark. The window is
   `int(fps / lfc × 2 + 1)` rounded up to odd: **7 frames (175 ms)** at 40 fps with `--lfc 12`.
   - Before filtering, **all** gaps are filled by linear interpolation (constant value at the start/end of the
     trial); after filtering, NaN is put back only for gaps of **6 frames or more**. Gaps of up to 5 frames
     (125 ms) stay filled.
   - A landmark with fewer valid frames than the window is not filtered.
   - ATHENA's other refinements (low-pass on the hand centroid, bone-length constraints) are **switched off** in
     this step.

So the final 3D can contain: values filled in by linear interpolation (constant speed, zero acceleration inside
the gap), values next to a gap distorted by the fill, and spikes from wrong detections that are attenuated but
not removed.

**Reliability of each point.** The script repeats the triangulation without smoothing and saves, for every frame
and landmark, the metrics in `analysis/<trial>/<trial>_3Dquality.npz`. Each point gets a class:

| class | meaning |
|---|---|
| **reliable** | triangulated (≥ 2 cameras) and passes all the checks below |
| **suspect** | triangulated, but fails at least one check |
| **interpolated** | seen by < 2 cameras, value filled in by the smoothing (gap of up to 5 frames) |
| **missing** | no value in the final 3D (gap of 6 or more frames) |

| check | fails when (default threshold) | what it reveals |
|---|---|---|
| `few_cams` | fewer than **3** cameras see the point | no redundancy: with 2 cameras a wrong detection cannot be detected |
| `reproj` | max reprojection error over the cameras > **20 px** | a camera whose 2D detection disagrees with the 3D point (wrong detection / wrong hand) |
| `outlier` | \|raw − final\| > **10 mm** | spike in the raw triangulation flattened by the smoothing |
| `near_gap` | within half a smoothing window (3 frames) of a gap | value partly distorted by the linear fill across the gap |

The defaults (`QUALITY_DEFAULTS` in `athena_analysis.py`) come from the distributions on the first recordings:
20 px and 10 mm are about the 95th–99th percentile. The QC uses them for `qc_summary.json` and for the summary
printed at the end of each run (`reliable … suspect … (<3 cams …, reproj …, outlier …, near gap …)`); the notebook
lets you change them interactively.

```python
import numpy as np, sys
sys.path.insert(0, 'scripts'); import athena_analysis as AA
q = dict(np.load('data/<recording>/athena/analysis/<trial>/<trial>_3Dquality.npz'))
cls, reasons = AA.quality_classes(q, min_cams=3, max_reproj_px=20, max_dev_mm=10)   # (frame, 75)
reliable = cls == 3                       # 0 missing, 1 interpolated, 2 suspect, 3 reliable
reproj_bad = reasons['reproj']            # also 'few_cams', 'outlier', 'near_gap'
```

For recordings processed before these metrics existed: `python scripts/athena_analysis.py data --skip-3d --redo`
(redoes only export and QC, a few seconds each). Alternatively the notebook computes them on first load.

### Joint angles – `scripts/hand_angles.py`

Computed from the final 3D for both hands, exported in `analysis/<trial>/<trial>_joint_angles.csv` and available in
the notebook (quantity *joint angles*). Degrees; names start with `R` / `L`.

| angle | definition | sign |
|---|---|---|
| **MCP flexion** (index, middle, ring, pinky) | metacarpal (wrist → MCP) vs proximal phalanx (MCP → PIP), around the hand flexion axis (index MCP → pinky MCP) | 0 = straight, + flexion, − hyperextension |
| **PIP flexion** | proximal phalanx vs middle phalanx, same axis | idem |
| **DIP flexion** | middle phalanx vs distal phalanx, same axis | idem |
| **TAM** (total active motion) | MCP + PIP + DIP of a finger | |
| **spread** index–middle, middle–ring, ring–pinky | angle between the proximal phalanges of adjacent fingers, on the palm plane (MCP abduction) | ≥ 0 |
| **thumb MCP / IP flexion** | consecutive thumb segments (CMC → MCP → IP → tip) | 0 = straight, ≥ 0 |
| **thumb abduction** | 1st (thumb) vs 2nd (index) metacarpal | ≥ 0 |
| **wrist flexion/extension** | forearm (elbow → wrist) vs hand (wrist → middle MCP), around the flexion axis | + palmar flexion, − extension |
| **wrist radial/ulnar deviation** | same vectors on the palm plane | + radial (thumb side), − ulnar |
| **forearm rotation** | palm normal vs vertical, around the forearm axis (world Z up) | 0 palm down … 90 vertical … 180 palm up |

Why these: MCP and PIP flexion are the main measures of finger function (grasp, tapping); DIP completes the finger
chain and TAM summarizes it (a standard clinical measure); spread describes the opening of the hand; thumb
abduction and thumb flexions describe grasp and opposition (thumb–finger distances are in the notebook as
*distance A–B*); the wrist and forearm angles describe the posture in which the hand works.

Notes:
* the flexion sign is the same for both hands (the flexion axis is mirrored for the left hand); verified on the
  recordings: PIP and MCP are almost always positive, DIP rarely slightly negative (physiological hyperextension);
* MediaPipe's thumb landmark 1 (CMC) is an approximation of the trapeziometacarpal joint: the thumb CMC flexion /
  opposition is not computed as a joint angle;
* the **wrist and forearm angles use the elbow from MediaPipe Pose**: on the current recordings its reprojection
  error is large (median 30–450 px, against 5–9 px for the hand), so they are classed *suspect* and should be
  treated as indicative;
* the quality of an angle in each frame (notebook) is the worst class among the landmarks it uses.

```python
import sys, numpy as np; sys.path.insert(0, 'scripts')
from hand_angles import hand_angles
angles, used = hand_angles(np.load('.../<trial>_3Dlandmarks.npy'), 'R')   # {name: (frames,) °}, {name: [landmarks]}
```

### Quality plots (`analysis/<trial>/`)

| File | What to look at |
|---|---|
| `01_acquisition.png` | frame intervals per camera (must be 25 ms at 40 fps), dropped frames, cross-camera skew. From the PTP timestamps in `Metadata_*.json` |
| `02_detection_2d.png` | per camera and over time: where MediaPipe detected right hand, left hand, body (% in the label) |
| `03_coverage.png` | number of cameras seeing each hand; red = missing in final 3D; blue = gap **filled by smoothing** (raw 3D was missing) |
| `04_gaps.png` + `gaps.csv` | timeline and histogram of gap durations, in raw and final 3D; the CSV has start/end/duration of every gap |
| `05_timeseries_<hand>.png` | x, y, z of wrist / thumb tip / index tip (line = final, dots = raw triangulation); speed with spikes (× = above median + 6·MAD); thumb–index distance; cameras per point |
| `06_reprojection.png` | reprojection error [px] per camera and over time: cameras with high error = wrong calibration or detections |
| `07_bone_lengths.png` | bone lengths over time: should be constant; high variability = noise or hand swaps |
| `qc_summary.json` | all numeric metrics (detection %, gaps, errors, spikes, bone CV) and, per hand, % reliable / suspect / interpolated / missing with the share of each failed check (`quality_thresholds` = thresholds used) |
| `run_options.json`, `run_info.json` | options and info/timings of this run |

Interpretation notes:
* *raw 3D* = direct triangulation of the detections (≥ 2 cameras needed). *final 3D* = ATHENA output after
  smoothing, which **interpolates gaps of up to 5 frames**: a "valid" value in the final data may be interpolated.
* Speed spikes during tapping may be real movements: evaluate them together with the
  thumb–index distance trace.

### ATHENA visual outputs

* `videos_processed/<trial>/data3d.mp4` – 3D skeleton (**always** produced)
* `videos_processed/<trial>/camN.mp4` – MediaPipe 2D detections (`standard`, `full` profiles)
* `videos_processed/<trial>/camN_refined.mp4` – 3D reprojected on each camera (`standard`, `full` profiles)
* per-frame images the videos are made from: `images/<trial>/camN/*.png`, `imagesrefined/<trial>/{camN,data3d}/*.jpg`
  (only `full` profile or `--keep-frames`; otherwise deleted once the videos are written)
* side-by-side video of all cameras: `athena-montage` (or `python -m athena.montage`, needs the images)

### Outputs and disk space

| `--outputs` profile | What is kept besides landmarks, analysis/QC and `data3d.mp4` | `athena/` for 46 s, 9 cameras |
|---|---|---|
| `minimal` | nothing (no overlay videos) | ~0.13 GB |
| `standard` (default) | overlay videos `camN.mp4` and `camN_refined.mp4` | ~0.3 GB |
| `full` | as `standard` + all per-frame images | ~4.7 GB |

Before these changes the same recording used ~9.3 GB in `athena/` (video copies + images).
With every profile the original `Take_*.mkv` files are untouched and `athena/videos` only holds links.
`minimal` is also faster (no 2D and reprojected overlays: saves writing ~1 image per camera per frame);
the 3D skeleton (`data3d.mp4`) remains the slowest part of the 3D step.

If a hand has a landmark that is **never** triangulated in the whole trial, ATHENA cannot draw the 3D
skeleton: `data3d.mp4` is skipped with a warning (`data3d_video = False` in the log).

**Cleaning up recordings already processed** (no re-analysis):

```bash
python scripts/athena_analysis.py data --cleanup --dry-run          # show what would be removed and how much
python scripts/athena_analysis.py data --cleanup                    # apply the standard profile
python scripts/athena_analysis.py data --cleanup --outputs minimal  # keep only data, QC and data3d.mp4
```

`--cleanup` (on one recording or the whole `data/`):
* replaces the mp4 copies in `athena/videos/` with links to `Take_*.mkv`, after checking they have the same
  number of frames;
* deletes the per-frame images (if a video required by the profile is missing, it is first created from them);
* with `minimal` also deletes the overlay videos `camN*.mp4`; `data3d.mp4` is never deleted
  (if it is missing, the message suggests re-running with `--skip-2d`).

Difference between `--cleanup` and `--cleanup --outputs minimal`: both turn the video copies into links and
delete the per-frame images (~9 GB per recording); only `videos_processed/<trial>/` differs:

| | `--cleanup` (standard) | `--cleanup --outputs minimal` |
|---|---|---|
| `data3d.mp4` – 3D skeleton | kept | kept |
| `camN.mp4` – original video with the points found by MediaPipe | kept | deleted |
| `camN_refined.mp4` – video with the final 3D reprojected | kept | deleted |

The 18 overlay videos take ~150–170 MB per recording and are used to visually check the tracking quality in
each camera: keeping them is recommended (plain `--cleanup`). If deleted, they are regenerated by re-running
the analysis (`--skip-2d` redoes only the `_refined` ones; `camN.mp4` also needs the 2D step).

### Data on a remote PC (SSH)

If the recordings are on another PC without a suitable GPU, the processing runs on this PC over SSH, without
keeping a permanent copy of the data. Required bandwidth: ~20 MB/s on average, because each mkv is read about
twice, once in the 2D step and once for the `_refined` overlays.

**A. Remote folder mounted with SSHFS** (recommended on a gigabit LAN). Nothing is copied: the script reads the
mkv files over the network and writes the results directly into `athena/` on the remote PC.

```bash
sudo apt install sshfs                                   # once
mkdir -p ~/remote_data
sshfs user@remote-pc:/path/data ~/remote_data -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

conda activate athena
python scripts/athena_analysis.py ~/remote_data --list              # what is on the remote PC and what is to do
python scripts/athena_analysis.py ~/remote_data                     # all those still to do
python scripts/athena_analysis.py ~/remote_data --select roberto --select 2026-11-12   # only some

fusermount -u ~/remote_data                              # unmount when done
```

If the connection drops, the current recording fails and the batch moves on to the next one; re-running the same
command skips the recordings already completed.

**B. Temporary copy, one recording at a time** (slow or unstable network, VPN, internet). Each recording is
copied locally (~5 GB), processed and deleted; only `athena/` goes back to the remote PC.

```bash
HOST=user@remote-pc
RDIR=/path/data
TMP=~/athena_tmp && mkdir -p $TMP
conda activate athena
SEL='*'                       # recordings to process: wildcard, e.g. '*roberto*' or '2026-11-12_*'
ssh $HOST "cd $RDIR && ls -d $SEL"          # check the selection first
for rec in $(ssh $HOST "cd $RDIR && ls -d $SEL"); do
    # skip recordings already completed on the remote PC
    ssh $HOST "ls $RDIR/$rec/athena/analysis/*/run_info.json" >/dev/null 2>&1 && continue
    rsync -a --info=progress2 --exclude 'athena/images' --exclude 'athena/imagesrefined' \
          "$HOST:$RDIR/$rec" "$TMP/" || continue
    python scripts/athena_analysis.py "$TMP/$rec"
    rsync -a "$TMP/$rec/athena/" "$HOST:$RDIR/$rec/athena/"      # results -> remote PC
    rm -rf "$TMP/$rec"                                            # no permanent copy
done
```

Notes:
* for an exact list instead of a wildcard: `for rec in 2026-11-10_111114_roberto_ninepeg_01 2026-11-10_111503_roberto_coin_01; do`;
* the run-time log `athena_runs_log.csv` stays on this PC; with SSHFS the timings include network reads;
* the `athena/videos/<trial>/cam_N.mp4` links are relative, so they stay valid on the remote PC too;
* passwordless access: `ssh-keygen`, then `ssh-copy-id user@remote-pc`.

### Collecting the 3D videos – `collect_data3d.py`

Copies every `videos_processed/<trial>/data3d.mp4` into a single `videos3ddata/` folder (in the project folder),
renamed after the recording: `videos3ddata/<recording>.mp4` (`<recording>__<trial>.mp4` if a recording has
more than one trial).

```bash
python scripts/collect_data3d.py                       # searches data/
python scripts/collect_data3d.py ~/remote_data         # another folder, e.g. the SSHFS mount
python scripts/collect_data3d.py --select roberto      # only some recordings
python scripts/collect_data3d.py --mode symlink        # links instead of copies (0 bytes, needs the data folder)
python scripts/collect_data3d.py --out /path/folder
```

Videos already collected and unchanged are skipped: re-run it after processing new recordings.

---

## 5. Run-time log and duration estimate – `athena_runs_log.csv`

Every run of `athena_analysis.py` appends a row (even if it fails: `status` = `error`/`interrupted`).
A copy of the row is in `analysis/<trial>/run_info.json`.

| Group | Columns |
|---|---|
| run | `run_start`, `status`, `error`, `recording`, `trial`, `steps` (2d/3d/qc executed) |
| video | `n_cameras`, `width`, `height`, `fps`, `codec`, `bitrate_mbps_per_cam`, `videos_total_mb`, `video_frames`, `video_duration_s` |
| options | `fraction`, `processed_frames`, `processed_duration_s`, `processed_camera_frames`, `use_gpu`, `processes`, `hand_conf`, `pose_conf`, `lfc_hz`, `save_*_images_video`, `outputs_profile`, `keep_frames`, `video_link` |
| timings [s] | `t_prepare_s`, `t_2d_s`, `t_3d_s` (includes images/videos), `t_qc_export_s`, `t_total_s` |
| throughput | `s_per_camera_frame_2d`, `fps_per_camera_2d`, `s_per_frame_3d`, `realtime_factor_total` (= compute time / processed duration) |
| extracted data | `right/left_hand_valid_pct`, `right/left_hand_gaps`, `n_images_written` (images rendered, even if deleted afterwards), `n_videos_written`, `data3d_video`, `output_*_mb` |
| machine | `hostname`, `cpu_model`, `cpu_count`, `ram_gb`, `gpu`, `mediapipe_version`, `athena_version` |

**A-priori estimate** for a new video (same machine and same options):

```
t_2d ≈ s_per_camera_frame_2d × n_frames × n_cameras
t_3d ≈ s_per_frame_3d × n_frames                     (dominated by images/videos if enabled)
t_total ≈ realtime_factor_total × video_duration_s
disk space ≈ output_images_videos_mb / processed_frames × n_frames
```

```python
import pandas as pd
log = pd.read_csv('athena_runs_log.csv')
ok = log[(log.status == 'ok') & (log.steps == '2d+3d+qc')]
print(ok.groupby(['outputs_profile', 'use_gpu'])[['s_per_camera_frame_2d', 's_per_frame_3d',
                                                   'realtime_factor_total']].median())
```

---

## 5b. Interactive visualization – `notebooks/landmarks3d_viewer.ipynb`

Jupyter notebook for exploring the 3D tracks and their quality. The controls choose:
* **recording**;
* **landmarks**, filtered by group: right hand, left hand, body;
* **quantity**:
  - x/y/z position;
  - speed (magnitude or per component) and acceleration;
  - A–B distance and A–B–C angle, e.g. finger flexion;
  - **joint angles** (MCP, PIP, DIP flexion, TAM, spread, thumb, wrist, forearm), chosen in the *Joint angles* list
    that replaces the landmark list; the *Statistics* tab then shows, per angle, quality and range of motion (ROM)
    on all frames and on the reliable frames only;
  - **bone length** ending at each selected landmark (e.g. `R_index_tip` = distal phalanx): it should be constant;
  - quality: number of cameras, **reprojection error** (max / mean over the cameras), **raw–final deviation**,
    with the threshold drawn as a dashed line;
* **time interval**;
* **quality thresholds** (panel): minimum cameras, maximum reprojection error, maximum raw–final deviation,
  frames near a gap. All plots and tables update.

The four tabs:

| Tab | Content |
|---|---|
| Time series | zoomable plots; bands **purple = suspect**, **orange = interpolated**, **red = missing**; the hover shows class and failed checks; optional raw triangulation overlay (position, bone length) |
| 3D trajectory | paths over time + hand skeleton at a chosen frame, joints coloured by class (hover: cameras, reprojection error, deviation) |
| Quality | map over time, per landmark, of class, number of cameras, reprojection error or raw–final deviation |
| Statistics | per landmark: % reliable/suspect/interpolated/missing, % of each failed check, mean cameras, median reprojection error, deviation p99, gaps, movement range, speed |

*Save plot* writes the interactive HTML and *Save data* writes the CSV with, for every landmark, the class
(`status_<landmark>`) and the failed checks (`flags_<landmark>`, e.g. `few_cams;reproj`) of each point, both in
`exports/`. The last cells of the notebook show a reliability table of all recordings.

```bash
~/miniforge3/envs/athena/bin/jupyter lab notebooks/landmarks3d_viewer.ipynb
```

In VSCode: open the notebook → *Select Kernel* → *Jupyter Kernel* → **Python (athena)**.

Everything you may want to change (data folder, initial recording, landmarks and quantity, reliability thresholds,
moving average, plot height, export folder) is in the **Parameters** cell. The same functions (`load_trial`, `timeseries_figure`,
`compute_series`, `stats_table`, ...) can also be called from code: see the last section of the notebook.

---

## 6. Original ATHENA GUI (optional)

```bash
python scripts/athena_gui.py        # GUI with fonts enlarged x2 (HiDPI screens / Wayland)
python scripts/athena_gui.py 1.5    # other factor
```

In the GUI select `data/<recording>/athena` as main folder. Note: the GUI defaults
(confidence 0.9) lose many hand detections in these recordings, and the per-frame 3D skeleton
blocks the interface for a long time. For normal use prefer `athena_analysis.py`.

---

## 7. Known issues

| Problem | Cause / solution |
|---|---|
| `cannot import name 'solutions' from 'mediapipe'` | `pip install "mediapipe==0.10.21"` |
| `No module named 'moviepy.editor'` | `pip install "moviepy<2"` |
| VSCode: the notebook controls cell stays "running" and the controls do not appear (Jupyter log: *Widget Error: Failed to access CDN*) | VSCode is not allowed to download the widget code: the project's `.vscode/settings.json` already has `"jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"]`; reload the window (*Developer: Reload Window*) and restart the kernel. Alternatively use JupyterLab |
| VSCode: `Import "numpy" could not be resolved`, or `python` not finding the packages | VSCode uses the old `.venv` (Python 3.14, empty) instead of the conda environment: delete `.venv` and pick the interpreter `~/miniforge3/envs/athena/bin/python` (*Python: Select Interpreter*) |
| `conda: command not found` after installation | `~/miniforge3/bin/conda init bash && source ~/.bashrc` |
| tiny text in the GUI | use `scripts/athena_gui.py` |
| hands detected in few cameras | the cameras frame the whole room: hands are small (60–100 px). Lower `--hand-conf`; check `02_detection_2d.png` |
| wrong 3D scale | wrong `--square-mm` in calibration: redo the calibration and then `athena_analysis.py` (`--skip-2d` is enough: 2D does not depend on scale) |
| "ERROR: no calibration" | run `athena_calibration.py` first |
| number of videos ≠ number of calibrations | ATHENA requires one calibration file per video in the trial |
