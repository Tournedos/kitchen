# kitchen

Tools for the 9-camera kitchen recordings (Dexter project): from the recorder's capture folders
to 3D body and hand landmarks with [ATHENA](https://github.com/neural-control-and-computation-lab/athena).

**Overview: [Kitchen Rig Calibration](https://claude.ai/artifact/NbkrzxvdhY8yXkuDrEXbMv)**, how the rig
was calibrated, how it follows ATHENA's README, and the checks showing it works (first capture,
2026-10-20). The page is private to its owner; share it from its Share menu. Its source is in
[docs/calibration-overview/](docs/calibration-overview/) (`facts.json` holds the numbers it quotes).
To change it, ask Claude to update that link: the link stays the same across versions.

## Pipeline

Kernel / environment: the `athena` conda env. Copy new capture folders into `Dexter/data/`, then run:

| Notebook | ATHENA README step | What it does |
|---|---|---|
| [1_prepare_data.ipynb](1_prepare_data.ipynb) | 1. Organize videos, 2. Ensure calibration is correct | converts the videos, computes the calibration from the checkerboard photos, checks it with ATHENA's own reader |
| [2_run_athena.ipynb](2_run_athena.ipynb) | 3. Processing (the GUI's GO button), 4. Montage | 2D detection, triangulation and smoothing, videos, montage, then a look at the 3D results |

Both skip work that is already done. The code behind them:

- [prepare_athena.py](prepare_athena.py): videos, calibration, checks (`python prepare_athena.py` does notebook 1 from a terminal)
- [calibrate_rig.py](calibrate_rig.py): calibration from the checkerboard photos
- [run_athena.py](run_athena.py): ATHENA's processing without its GUI

## Step 1: organize the videos

For every capture folder in `Dexter/data/` (any folder with the recorder's `metadata.json`):

```
Dexter/athena_projects/<capture>/
├── videos/<capture>/cam0.avi ... cam8.avi   lossless remux of Take_*.mkv (same frames, AVI container)
├── calibration/calibration.toml              camera calibration, computed from the checkerboards
├── calibration_check.jpg                     floor grid drawn into every camera (visual check)
├── calibration_report.json                   per-camera errors, focal lengths, positions
├── preview.jpg                               one labelled frame per camera
└── prep_manifest.json                        camera map + recorder checks
```

ATHENA pairs videos and calibration entries by sorted filename and names all its outputs
`cam0..cam8`. The lab names cameras `cam_1..cam_9`:

| ATHENA | lab   | serial   |
|--------|-------|----------|
| cam0   | cam_1 | 25369878 |
| cam1   | cam_2 | 25369880 |
| cam2   | cam_3 | 25369858 |
| cam3   | cam_4 | 25369877 |
| cam4   | cam_5 | 25369887 |
| cam5   | cam_6 | 25369868 |
| cam6   | cam_7 | 25369867 |
| cam7   | cam_8 | 25369888 |
| cam8   | cam_9 | 25369859 |

The mapping is read from each capture's `metadata.json` and written to `prep_manifest.json`.

## Step 2: calibration

ATHENA needs each camera's lens (focal length, distortion) and position to turn 2D detections into
3D. At the start of every capture the recorder saves one photo per camera of the two checkerboards
(7x5 squares of 10 cm) on the floor and the counter (`calibration_frames/*/cam_N_frame1.png`).
`calibrate_rig.py`:

1. detects the boards in every photo;
2. works out which board is which and its orientation (the pattern looks the same rotated by 180°),
   by fitting every possibility and keeping the only one that is geometrically consistent;
3. jointly optimizes all cameras, using the board corners plus SIFT feature matches between cameras
   (all photos are taken at the same instant, so the kitchen itself is a calibration object);
4. puts the origin on the corner of the floor board, z pointing up, units in mm.

The calibration file names cameras `cam0..cam8`, matching the video files, as ATHENA's README
requires; `check_project` loads it with ATHENA's reader and checks it against the videos.

Checks on the first capture: all cameras below 1 px reprojection error; the floor grid in
`calibration_check.jpg` lies flat in every view; the board leaning on a cabinet later in the video
(never used for calibration) reconstructs as 100 ± 0.3 mm squares, flat within 1 mm; the person's
nose comes out ~1.56 m above the floor and the ankles ~0.1 m.

Limits: it needs both boards visible together in at least one camera, and at least one board in
every camera. A calibration video with a board moved around the room would be more accurate at the
image edges than a single snapshot.

To use a calibration made elsewhere instead, put it in `data/<capture>/calibration/` (or pass
`--calibration DIR`): JARVIS `.yaml` per camera named by lab name (`cam_1.yaml`) or serial, or one
Anipose `.toml` whose `name` fields are lab names or serials.

## Steps 3 and 4: run ATHENA

The GUI only collects options and calls ATHENA's two phases; `run_athena.py` calls the same
functions with the same option names, so the notebook needs no GUI:

- **phase 1** (`detect_2d`): MediaPipe body (33 points) and hands (21 points each) on every
  undistorted frame of every camera, then left/right hands checked across cameras;
- **phase 2** (`triangulate_3d`): hand swaps corrected, 3D triangulation with outlier cameras
  dropped, smoothing, reprojection, videos;
- **montage** (`make_montage`): ATHENA's `athena-montage` uses the moviepy 1.x API, which the
  installed moviepy 2.x no longer has, so this redoes it with the same layout.

Options changed from the GUI defaults, measured on this kitchen:

- **pose and hand confidence 0.5 instead of 0.9.** At 0.9 the body is found in 0–3 of 7 test frames
  per camera; at 0.5 in most frames, and both wrists triangulate in ~93% of frames.
- **3D and reprojected videos on**, which the montage needs.

Timing for the 103 s first capture on this Mac: phase 1 ≈ 5 min; phase 2 ≈ 75 min, almost all of
it drawing the reprojected 2D images (≈ 2 min with `save_images_refine=False`, but then no
montage); montage ≈ 10 min.

Results on the first capture: 3D body in 97% of frames, hands in ~50%; median nose height 1.53 m,
ankles 0.06 m. **Frames seen by fewer than 4 cameras are unreliable**: when the person walked out
at the end (2–3 cameras left), the triangulation jumped by metres between frames. The notebook's
`clean` array sets those frames (7% here) to NaN; `run_athena.cameras_seeing` gives the count.

**Hands are the weak point.** The cameras see the whole room, so a hand is ~40 px wide in a 1920 px
frame; MediaPipe's hand detector, which looks at the whole frame shrunk to 192 px, finds hands in
only ~0–15% of frames per camera. Wrists and arms come from the body model and are fine. For
fingers, ATHENA's `hamer` backend looks for hands around the body's wrists (much better at this
scale) but is much slower and needs the MANO hand model: register at https://mano.is.tue.mpg.de,
download it, and put `MANO_RIGHT.pkl` where `run_athena.hamer_missing()` says. Face mesh needs
MediaPipe's `face_landmarker.task`, which this ATHENA install does not include.

Outputs, in `Dexter/athena_projects/<capture>/`:

- `landmarks/<rec>/<rec>_3Dlandmarks.npy`: `(frames, 75, 3)` in mm (0–32 body, 33–53 right hand,
  54–74 left hand; NaN where fewer than 2 cameras saw the point)
- `landmarks/<rec>/<rec>_2Dlandmarksrefined.npy`: the 3D points reprojected into each camera
- `videos_processed/<rec>/`: `cam*_refined.mp4`, `data3d.mp4`, `compilation.mp4`
