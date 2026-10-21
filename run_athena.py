"""Run ATHENA's processing without its GUI (ATHENA README steps 3 and 4).

The GUI (athena/athena.py) only gathers options and then calls
labels2d.main (phase 1, 2D detection) and triangulaterefine.main (phase 2,
3D). Here:

- detect_2d makes the same calls as labels2d.main's processing loop, but
  reports progress in the notebook instead of a Tk window that blocks until done
- triangulate_3d calls triangulaterefine.main directly (it has no GUI)
- make_montage redoes athena.montage with the same layout: the original asks
  for the folder in a dialog and uses moviepy 1.x (moviepy.editor), which the
  installed moviepy 2.x no longer has

Options use the GUI's names and defaults (GUI_DEFAULTS).
"""

import glob
import json
import os
import queue
import threading
import time
from multiprocessing import Manager
from pathlib import Path

import av
import numpy as np

# Same option names and defaults as the GUI (athena/athena.py).
GUI_DEFAULTS = {
    # General / 2D extraction
    "fraction_frames": 1.0,          # fraction of each video to process (from the start)
    "num_processes": os.cpu_count(),  # cameras processed in parallel
    "use_gpu": False,
    "hand_confidence": 0.9,
    "pose_confidence": 0.9,
    "hand_backend": "mediapipe",     # or "hamer"
    "use_face_mesh": False,
    "save_images_mp": False,         # 2D overlay images ...
    "save_video_mp": False,          # ... and videos (needs images)
    # Triangulation & refinement
    "refine_calibration": False,
    "all_landmarks_lfc": 10,         # smoothing low-pass cutoff (Hz)
    "save_images_triangulation": False,  # 3D skeleton images ...
    "save_video_triangulation": False,   # ... and video (needed for the montage)
    "save_images_refine": False,         # reprojected 2D images ...
    "save_video_refine": False,          # ... and videos (needed for the montage)
}

# Landmark layout of <recording>_3Dlandmarks.npy (ATHENA README, "3D Landmark File Format").
BODY = slice(0, 33)
RIGHT_HAND = slice(33, 54)
LEFT_HAND = slice(54, 75)
NOSE, RIGHT_WRIST, LEFT_WRIST = 0, 33, 54


def list_recordings(project):
    return sorted(p.name for p in (Path(project) / "videos").iterdir() if p.is_dir())


def gui_options(project, recordings=None, **options):
    """The options dict the GUI would pass to ATHENA."""
    unknown = set(options) - set(GUI_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}; valid: {sorted(GUI_DEFAULTS)}")
    project = Path(project)
    opts = {**GUI_DEFAULTS, **options}
    opts["num_processes"] = max(1, min(opts["num_processes"], os.cpu_count()))
    # The GUI ticks "save images" whenever "save video" is ticked.
    for images, video in (("save_images_mp", "save_video_mp"),
                          ("save_images_triangulation", "save_video_triangulation"),
                          ("save_images_refine", "save_video_refine")):
        opts[images] = opts[images] or opts[video]
    opts["main_folder"] = str(project)
    opts["idfolders"] = [str(project / "videos" / r) for r in (recordings or list_recordings(project))]
    return opts


def _calibration_files(project):
    cal = Path(project) / "calibration"
    for ext in ("*.yaml", "*.toml"):  # same preference as ATHENA
        files = sorted(glob.glob(str(cal / ext)))
        if files:
            return files, ext
    raise FileNotFoundError(f"no calibration in {cal}")


def _fps(video):
    with av.open(video) as container:
        rate = container.streams.video[0].average_rate
        return float(rate) if rate else 30.0


class _Progress(threading.Thread):
    """Prints per-camera progress from ATHENA's progress queue every `every` seconds."""

    def __init__(self, progress_queue, ncams, every=20):
        super().__init__(daemon=True)
        self.queue, self.ncams, self.every = progress_queue, ncams, every
        self.progress, self.fps = {}, {}
        self.finished = threading.Event()
        self.start_time = time.time()

    def run(self):
        last = time.time()
        while not self.finished.is_set():
            try:
                msg = self.queue.get(timeout=1)
                cam = msg.get("cam")
                if cam is not None:
                    if "progress" in msg:
                        self.progress[cam] = msg["progress"]
                    if "fps" in msg:
                        self.fps[cam] = msg["fps"]
                    if msg.get("done"):
                        self.progress[cam] = 100.0
            except queue.Empty:
                pass
            except (EOFError, BrokenPipeError):
                break
            if time.time() - last > self.every:
                self.report()
                last = time.time()

    def report(self):
        done = sum(self.progress.get(c, 0) for c in range(self.ncams)) / self.ncams
        fps = np.mean(list(self.fps.values())) if self.fps else 0
        print(f"    {done:5.1f}% done after {(time.time() - self.start_time) / 60:.1f} min "
              f"({fps:.1f} frames/s per camera)", flush=True)

    def stop(self):
        self.finished.set()
        self.join()
        self.report()


def hamer_missing():
    """What is missing for the HaMeR hand backend (empty string if ready).

    ATHENA downloads several GB of HaMeR checkpoints before checking for the
    MANO hand model, which needs a manual, licensed download; check it first.
    """
    import athena
    from athena.hamer_hands import is_available

    if not is_available():
        return "HaMeR/PyTorch are not installed (pip install athena-tracking[hamer])"
    mano_dir = Path(athena.__file__).parent.parent / "_DATA" / "data" / "mano"
    if not (mano_dir / "MANO_RIGHT.pkl").exists():
        return ("the MANO hand model is missing: register at https://mano.is.tue.mpg.de, "
                f"download the MANO model and put MANO_RIGHT.pkl in {mano_dir}/")
    return ""


def detect_2d(project, recordings=None, **options):
    """Phase 1 ("Run 2D Extraction"): per-camera body/hand landmarks, then hand
    labels corrected across cameras. Writes landmarks/<recording>/cam*/."""
    from athena import labels2d

    opts = gui_options(project, recordings, **options)
    if opts["hand_backend"] == "hamer" and hamer_missing():
        raise RuntimeError("HaMeR backend not ready: " + hamer_missing())
    if opts["use_face_mesh"] and not os.path.exists(labels2d.face_model_path):
        raise RuntimeError(f"face mesh needs MediaPipe's face_landmarker.task at {labels2d.face_model_path} "
                           "(not shipped with this ATHENA install)")
    project = Path(project)
    calfiles, ext = _calibration_files(project)
    extrinsic, intrinsic, dist = labels2d.read_calibration(calfiles, ext)
    run = labels2d.run_hybrid if opts["hand_backend"] == "hamer" else labels2d.run_mediapipe

    for trial in sorted(opts["idfolders"]):
        name = os.path.basename(trial)
        videos = sorted(glob.glob(os.path.join(trial, "*.avi")) + glob.glob(os.path.join(trial, "*.mp4")))
        if len(videos) != len(intrinsic):
            raise ValueError(f"{name}: {len(videos)} videos but {len(intrinsic)} calibrated cameras")
        out_images = os.path.join(project, "images", name)
        out_data = os.path.join(project, "landmarks", name)
        os.makedirs(out_data, exist_ok=True)
        if opts["save_images_mp"]:
            for cam in range(len(videos)):
                os.makedirs(os.path.join(out_images, f"cam{cam}"), exist_ok=True)

        print(f"{name}: {len(videos)} cameras, {opts['hand_backend']} hands, "
              f"{opts['fraction_frames']:.0%} of frames, {opts['num_processes']} processes", flush=True)
        with Manager() as manager:
            progress = _Progress(manager.Queue(), len(videos))
            progress.start()
            try:
                run(videos, dict(opts), intrinsic, dist, out_images, out_data, name,
                    cam_mats_extrinsic=extrinsic, progress_queue=progress.queue)
            finally:
                progress.stop()

        with open(os.path.join(out_data, "detection_meta.json"), "w") as f:
            json.dump({"hand_backend": opts["hand_backend"]}, f)
        if opts["save_video_mp"]:
            suffix = "hamer" if opts["hand_backend"] == "hamer" else "mediapipe"
            for cam in range(len(videos)):
                labels2d.create_video(os.path.join(out_images, f"cam{cam}"), ".png", _fps(videos[0]),
                                      os.path.join(project, "videos_processed", name), f"cam{cam}_{suffix}.mp4")


def triangulate_3d(project, recordings=None, **options):
    """Phase 2 ("Run Triangulation and Refinement"): hand-swap correction, DLT
    triangulation, smoothing, reprojection. Writes
    landmarks/<recording>/<recording>_3Dlandmarks.npy and _2Dlandmarksrefined.npy."""
    from athena import triangulaterefine

    triangulaterefine.main(json.dumps(gui_options(project, recordings, **options)))


def make_montage(project, recordings=None):
    """README step 4 (athena-montage): every camera's refined 2D video plus the
    3D video, tiled into videos_processed/<recording>/compilation.mp4.

    Same layout as athena.montage; written at the recordings' own frame rate
    (athena.montage hard-codes 60 fps).
    """
    from moviepy import VideoFileClip, clips_array

    outputs = []
    for name in recordings or list_recordings(project):
        folder = Path(project) / "videos_processed" / name
        cams = sorted(folder.glob("*_refined.mp4"))
        video3d = folder / "data3d.mp4"
        if not cams or not video3d.exists():
            raise FileNotFoundError(f"{folder}: run triangulate_3d with save_video_refine=True "
                                    "and save_video_triangulation=True first")
        clips = [VideoFileClip(str(p)) for p in cams]
        bottom = clips_array([[VideoFileClip(str(video3d)).resized(1.5)]])
        if len(clips) <= 4:
            rows = [[clips_array([clips])]]
        else:
            half = round(len(clips) / 2)
            rows = [[clips_array([clips[:half]])], [clips_array([clips[half:]])]]
        out = folder / "compilation.mp4"
        clips_array(rows + [[bottom]]).write_videofile(str(out), codec="libx264", fps=clips[0].fps, logger=None)
        outputs.append(out)
    return outputs


def load_3d(project, recording):
    """(nframes, nlandmarks, 3) array in mm, world frame of the calibration."""
    return np.load(Path(project) / "landmarks" / recording / f"{recording}_3Dlandmarks.npy")


def cameras_seeing(project, recording):
    """Per frame: how many cameras detected the body (MediaPipe pose is all-or-nothing).

    With only 2-3 cameras the triangulation can jump around; on the first
    capture every jump of the hips > 0.3 m between frames had fewer than 4.
    """
    folder = Path(project) / "landmarks" / recording
    return np.sum([np.load(cam / "2Dlandmarks_body.npy")[:, 0, 0] != -1
                   for cam in sorted(folder.glob("cam*"))], axis=0)


def detection_rates(project, recording):
    """Per camera: fraction of processed frames with a body / right hand / left hand detected."""
    rows = {}
    for cam_dir in sorted((Path(project) / "landmarks" / recording).glob("cam*")):
        rates = {}
        for part in ("body", "right", "left"):
            data = np.load(cam_dir / f"2Dlandmarks_{part}.npy")
            rates[part] = float(np.mean(data[:, 0, 0] != -1)) if len(data) else 0.0
        rows[cam_dir.name] = rates
    return rows
