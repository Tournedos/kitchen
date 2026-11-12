#!/usr/bin/env python3
"""
ATHENA pipeline (headless) + data-quality analysis for one recording.

    python scripts/athena_analysis.py [FOLDER] [options]

FOLDER can be either
    * a recording folder  (data/2026-11-10_112048_louis_finertapping_01)  -> main folder = FOLDER/athena
    * an ATHENA main folder (contains calibration/ and videos/)
If omitted, a folder-selection dialog opens.

Steps
    1. prepare   : link Take_*.mkv -> athena/videos/<trial>/cam_N.mp4 (relative symlink by default, no extra
                   space; --video-link hardlink|copy), mapping from calibration_report.json (athena_calibration.py)
    2. ATHENA 2D : athena.labels2d (MediaPipe hands + pose on every camera), without the Tk progress window
    3. ATHENA 3D : athena.triangulaterefine (hand switching, triangulation, smoothing, reprojection)
    4. export    : 3D landmarks as CSV (+ hands-only npy)
    5. QC        : plots + summary in athena/analysis/<trial>/
                     01_acquisition.png      frame intervals / skips / cross-camera skew (Metadata_*.json)
                     02_detection_2d.png     per-camera detection timeline (right, left, body)
                     03_coverage.png         cameras per frame, raw vs final 3D validity
                     04_gaps.png             gap timeline + gap length histogram
                     05_timeseries_<hand>.png  wrist / thumb tip / index tip xyz, speed + spikes, thumb-index distance
                     06_reprojection.png     reprojection error per camera and over time
                     07_bone_lengths.png     bone length variability (should be ~constant)
                     qc_summary.json, gaps.csv

Outputs (--outputs PROFILE; landmarks, analysis/QC and videos_processed/<trial>/data3d.mp4 are always written):
    minimal   : + nothing else (no 2D / refined overlay videos)
    standard  : + videos_processed/<trial>/camN.mp4 (2D MediaPipe overlay) and camN_refined.mp4 (reprojected 3D)
                (default)  per-frame images are deleted once the videos are written
    full      : as standard, and the per-frame images are kept:
                athena/images/<trial>/camN/*.png, athena/imagesrefined/<trial>/{camN,data3d}/*.jpg
    --keep-frames keeps the per-frame images with any profile.

Storage cleanup of recordings already processed (no re-analysis):
    python scripts/athena_analysis.py data --cleanup --dry-run     # show what would be removed
    python scripts/athena_analysis.py data --cleanup [--outputs minimal]
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # before any matplotlib/athena import (also in spawned workers)

import argparse
import csv
import glob
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


def patch_matplotlib_line3d():
    """matplotlib >= 3.11: Line3D.draw crashes ('list' object has no attribute 'shape') when a line
    contains NaN and its 3D vertices are Python lists, which is what ATHENA's visualize_3d passes.
    Convert the vertices to float arrays before drawing. Harmless on older matplotlib versions."""
    from mpl_toolkits.mplot3d import art3d
    if getattr(art3d.Line3D.draw, '_athena_patched', False):
        return
    orig_draw = art3d.Line3D.draw

    def draw(self, renderer):
        self._verts3d = tuple(np.asarray(v, dtype=float) for v in self._verts3d)
        return orig_draw(self, renderer)

    draw._athena_patched = True
    art3d.Line3D.draw = draw

# --------------------------------------------------------------------------------------
# Landmark definitions (ATHENA 3D order: 0-32 body, 33-53 right hand, 54-74 left hand)
# --------------------------------------------------------------------------------------
HAND_NAMES = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
              'index_mcp', 'index_pip', 'index_dip', 'index_tip',
              'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
              'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
              'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
BODY_NAMES = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
              'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
              'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
              'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
              'left_foot_index', 'right_foot_index']
LANDMARK_NAMES = (['body_' + n for n in BODY_NAMES] + ['R_' + n for n in HAND_NAMES] + ['L_' + n for n in HAND_NAMES])
HANDS = {'right': 33, 'left': 54}
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
KEY_POINTS = {'wrist': 0, 'thumb_tip': 4, 'index_tip': 8}
DATE_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{6}_')


def log(msg):
    print(f'[athena_analysis] {msg}', flush=True)


# --------------------------------------------------------------------------------------
# 0. Folder resolution
# --------------------------------------------------------------------------------------

def resolve_folders(folder):
    folder = folder.resolve()
    if (folder / 'calibration').is_dir() and (folder / 'videos').is_dir():
        main = folder
    elif (folder / 'calibration').is_dir():
        main = folder
    else:
        main = folder / 'athena'
    rec_dir = main.parent if list(main.parent.glob('Take_*')) or list(main.parent.glob('Metadata_*.json')) else None
    if not (main / 'calibration').is_dir() or not (glob.glob(str(main / 'calibration' / '*.yaml')) or
                                                 glob.glob(str(main / 'calibration' / '*.toml'))):
        sys.exit(f'ERROR: no calibration in {main}/calibration. Run athena_calibration.py first.')
    return main, rec_dir


def ncalib(main):
    yamls = sorted((main / 'calibration').glob('*.yaml'))
    if yamls:
        return len(yamls)
    import toml
    return len(toml.load(next((main / 'calibration').glob('*.toml')))) - 1


def default_trials(main, rec_dir):
    vdir = main / 'videos'
    if vdir.is_dir():
        subs = sorted(d.name for d in vdir.iterdir() if d.is_dir())
        if subs:
            return subs
    if rec_dir is None:
        sys.exit(f'ERROR: no videos in {vdir} and no source recording to convert from.')
    return [DATE_PREFIX.sub('', rec_dir.name) or rec_dir.name]


# --------------------------------------------------------------------------------------
# 1. Video preparation
# --------------------------------------------------------------------------------------

def find_ffmpeg():
    for c in (shutil.which('ffmpeg'), str(Path(sys.prefix) / 'bin' / 'ffmpeg')):
        if c and Path(c).exists():
            return c
    sys.exit('ERROR: ffmpeg not found. Install it with: conda install -c conda-forge ffmpeg')


def video_frames(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = s.frames
        if not n:
            n = sum(1 for _ in c.demux(s) if _.size)
        fps = float(s.average_rate) if s.average_rate else 30.0
    return int(n), fps


def video_info(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return {'width': s.codec_context.width, 'height': s.codec_context.height, 'codec': s.codec_context.name,
                'bitrate_mbps': round(Path(path).stat().st_size * 8 / max(float(c.duration or 1) / av.time_base, 1e-9)
                                      / 1e6, 2)}


def video_mapping(main, rec_dir):
    """[(source mkv, athena video path)] from calibration_report.json (athena_calibration.py)."""
    report = main / 'calibration_report.json'
    if not report.exists():
        sys.exit(f'ERROR: {report} missing (camera <-> video mapping). Run athena_calibration.py first.')
    return [(rec_dir / c['source_video'], Path(c['athena_video_name']).stem + '.mp4')
            for c in json.loads(report.read_text())['athena_order']]


def link_video(src, dst, mode):
    """Make dst point to src without duplicating data: relative symlink, hard link or (stream) copy."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == 'symlink':
        dst.symlink_to(os.path.relpath(src, dst.parent))  # relative: survives moving the whole recording folder
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        subprocess.run([find_ffmpeg(), '-loglevel', 'error', '-y', '-i', str(src), '-c', 'copy', str(dst)], check=True)


def is_linked(src, dst):
    """True if dst already refers to src without its own copy of the data."""
    if dst.is_symlink():
        return dst.exists() and dst.resolve() == src.resolve()
    return dst.exists() and src.exists() and os.path.samefile(src, dst)


def prepare_videos(main, rec_dir, trial, link_mode='symlink'):
    vdir = main / 'videos' / trial
    ncal = ncalib(main)
    vids = sorted(list(vdir.glob('*.mp4')) + list(vdir.glob('*.avi'))) if vdir.is_dir() else []
    vids = [v for v in vids if v.exists()]  # drop broken symlinks
    if len(vids) != ncal:
        if rec_dir is None:
            sys.exit(f'ERROR: {vdir} has {len(vids)} videos but there are {ncal} calibration files.')
        vdir.mkdir(parents=True, exist_ok=True)
        for src, name in video_mapping(main, rec_dir):
            dst = vdir / name
            if dst.exists():
                continue
            if not src.exists():
                sys.exit(f'ERROR: source video {src} not found')
            log(f'{link_mode} {src.name} -> {dst.relative_to(main)}')
            link_video(src, dst, link_mode)
        vids = sorted(v for v in vdir.glob('*.mp4') if v.exists())
    if len(vids) != ncal:
        sys.exit(f'ERROR: {len(vids)} videos vs {ncal} calibration files in {main}')
    counts = {}
    fps = None
    for v in vids:
        counts[v.name], fps = video_frames(v)
    if len(set(counts.values())) != 1:
        log(f'WARNING: videos have different frame counts: {counts}')
    log(f'{len(vids)} videos, {max(counts.values())} frames, {fps:g} fps')
    info = video_info(vids[0])
    info.update({'n_cameras': len(vids), 'n_frames': max(counts.values()), 'fps': fps,
                 'video_duration_s': round(max(counts.values()) / fps, 3),
                 'videos_total_mb': round(sum(v.stat().st_size for v in vids) / 1e6, 1)})
    return vids, fps, info


# --------------------------------------------------------------------------------------
# 2. ATHENA 2D (labels2d without Tk window)
# --------------------------------------------------------------------------------------

def run_athena_2d(main, trial, opts):
    import athena.labels2d as L
    from multiprocessing import Manager

    for sub in ('images', 'videos_processed'):
        shutil.rmtree(main / sub / trial, ignore_errors=True)
    calext = '*.yaml' if glob.glob(str(main / 'calibration' / '*.yaml')) else '*.toml'
    calfiles = sorted(glob.glob(str(main / 'calibration' / calext)))
    if calext == '*.toml':
        calfiles = calfiles[0]
    _, K, dist = L.readcalibration(calfiles, calext)

    trialdir = main / 'videos' / trial
    vidnames = sorted(glob.glob(str(trialdir / '*.avi')) + glob.glob(str(trialdir / '*.mp4')))
    out_img = main / 'images' / trial
    out_data = main / 'landmarks' / trial
    out_data.mkdir(parents=True, exist_ok=True)
    if opts['save_images_mp']:
        for cam in range(len(vidnames)):
            (out_img / f'cam{cam}').mkdir(parents=True, exist_ok=True)

    manager = Manager()
    q = manager.Queue()
    done = threading.Event()
    prog, fpsd = {}, {}

    def monitor():
        last = 0
        while not done.is_set() or not q.empty():
            try:
                m = q.get(timeout=0.2)
                if 'progress' in m and 'cam' in m:
                    prog[m['cam']] = m['progress']
                if 'fps' in m and 'cam' in m:
                    fpsd[m['cam']] = m['fps']
            except Exception:
                pass
            if time.time() - last > 2 and prog:
                last = time.time()
                p = sum(prog.values()) / len(vidnames)
                f = sum(fpsd.values()) / max(len(fpsd), 1)
                print(f'\r  2D detection: {p:5.1f}%   ({f:.1f} fps per camera)   ', end='', flush=True)
        print()

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.time()
    try:
        L.run_mediapipe(vidnames, dict(opts), K, dist, str(out_img), str(out_data), trial, progress_queue=q)
    finally:
        done.set()
        th.join()
    log(f'2D detection done in {time.time() - t0:.0f} s')

    n_img = count_files(out_img, '*.png')
    if opts['save_images_mp'] and opts['save_video_mp']:
        _, fps = video_frames(vidnames[0])
        for cam in range(len(vidnames)):
            L.createvideo(str(out_img / f'cam{cam}'), '.png', fps, str(main / 'videos_processed' / trial),
                          f'cam{cam}.mp4')
        log('2D overlay videos written')
    if not opts.get('keep_frames'):
        drop_frames(out_img)
    return n_img


def drop_frames(path):
    """Delete a folder of per-frame images (the videos made from them are kept)."""
    if Path(path).is_dir():
        mb = dir_mb(Path(path))
        shutil.rmtree(path, ignore_errors=True)
        log(f'per-frame images deleted ({mb:.0f} MB): {path}')


# --------------------------------------------------------------------------------------
# 3. ATHENA 3D (triangulaterefine.main)
# --------------------------------------------------------------------------------------

def run_athena_3d(main, trial, opts):
    import athena.triangulaterefine as T
    patch_matplotlib_line3d()
    trial_lm = main / 'landmarks' / trial
    camdirs = sorted(d for d in trial_lm.iterdir() if d.is_dir()) if trial_lm.is_dir() else []
    if len(camdirs) != ncalib(main):
        sys.exit(f'ERROR: {len(camdirs)} camera folders in {trial_lm}, expected {ncalib(main)}. Run the 2D step.')

    opts = dict(opts)
    if opts['save_images_triangulation']:
        # visualize_3d crashes if a hand is never triangulated (NaN axis limits): check beforehand
        raw = triangulate_raw(main, trial)[0]
        if not np.isfinite(raw[:, 33:74]).any(axis=0).all():  # every hand landmark seen at least once?
            log('WARNING: some hand landmarks are never triangulated -> 3D skeleton images/video '
                '(data3d.mp4) cannot be drawn and are skipped')
            opts['save_images_triangulation'] = False
            opts['save_video_triangulation'] = False
    vproc = main / 'videos_processed' / trial
    for f in [vproc / 'data3d.mp4'] + ([] if opts['save_video_refine'] else list(vproc.glob('cam*.mp4'))):
        f.unlink(missing_ok=True)  # no stale videos from previous runs with other options
    out_ref = main / 'imagesrefined' / trial
    shutil.rmtree(out_ref, ignore_errors=True)
    opts['main_folder'] = str(main)
    opts['idfolders'] = [str(main / 'videos' / trial)]
    t0 = time.time()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        T.main(json.dumps(opts))
    log(f'triangulation + outputs done in {time.time() - t0:.0f} s')
    n_img = count_files(out_ref, '*.jpg')
    if not opts.get('keep_frames'):
        drop_frames(out_ref)
    return n_img


# --------------------------------------------------------------------------------------
# Raw (unsmoothed) triangulation, using ATHENA's own functions -> used by QC
# --------------------------------------------------------------------------------------

def load_calibration(main):
    import athena.labels2d as L
    calext = '*.yaml' if glob.glob(str(main / 'calibration' / '*.yaml')) else '*.toml'
    calfiles = sorted(glob.glob(str(main / 'calibration' / calext)))
    if calext == '*.toml':
        calfiles = calfiles[0]
    ext, K, dist = L.readcalibration(calfiles, calext)
    return np.array(ext), K, dist


def load_2d(main, trial):
    trial_lm = main / 'landmarks' / trial
    camdirs = sorted(d for d in trial_lm.iterdir() if d.is_dir())
    R = np.stack([np.load(d / '2Dlandmarks_right.npy').astype(float) for d in camdirs])
    Lh = np.stack([np.load(d / '2Dlandmarks_left.npy').astype(float) for d in camdirs])
    B = np.stack([np.load(d / '2Dlandmarks_body.npy').astype(float) for d in camdirs])
    hs = []
    for d in camdirs:
        f = d / 'handedness_score.npy'
        hs.append(np.load(f) if f.exists() else np.full((R.shape[1], 2), -1.0))
    return R, Lh, B, np.stack(hs)


def triangulate_raw(main, trial):
    """Replicates triangulaterefine.main up to (excluded) smoothing. Returns raw3d, switched 2D, ncams/point."""
    import athena.triangulaterefine as T
    ext, K, _ = load_calibration(main)
    R, Lh, B, _ = load_2d(main, trial)
    comb = np.concatenate((B[..., :2], R[..., :2], Lh[..., :2]), axis=2)
    ncams, nframes, nlm, _ = comb.shape
    sw = T.switch_hands(comb.copy(), ncams, nframes, nlm, K, ext)
    sw = np.asarray(sw, dtype=float).reshape(ncams, nframes, nlm, 2)
    sw[(sw[..., 0] == -1) & (sw[..., 1] == -1)] = np.nan

    und = np.empty_like(sw)
    for c in range(ncams):
        pts = sw[c].reshape(-1, 2)
        und[c] = T.undistort_points(pts.astype(float), K[c], np.zeros(5)).reshape(nframes, nlm, 2)
    good = ~np.isnan(und[..., 0])  # ncams, nframes, nlm
    nc = good.sum(0)
    raw = np.full((nframes, nlm, 3), np.nan)
    P = ext[:, :3, :]
    for f in range(nframes):
        for p in np.nonzero(nc[f] >= 2)[0]:
            g = good[:, f, p]
            raw[f, p] = T.triangulate_simple(und[g, f, p], P[g])
    return raw, sw, nc


# --------------------------------------------------------------------------------------
# 4. Export
# --------------------------------------------------------------------------------------

def export(main, trial, d3, fps, outdir):
    path = outdir / f'{trial}_3Dlandmarks.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'time_s'] + [f'{n}_{a}' for n in LANDMARK_NAMES for a in 'xyz'])
        for f in range(d3.shape[0]):
            w.writerow([f, f'{f / fps:.4f}'] + [('' if np.isnan(v) else f'{v:.3f}') for v in d3[f].ravel()])
    np.save(outdir / f'{trial}_hands3d.npy', np.stack([d3[:, 33:54], d3[:, 54:75]], axis=1))  # (frames, 2[R,L], 21, 3)
    # joint angles of both hands (hand_angles.py): MCP/PIP/DIP flexion, TAM, spread, thumb, wrist, forearm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hand_angles as HA
    ang, _ = HA.all_hand_angles(d3)
    with open(outdir / f'{trial}_joint_angles.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'time_s'] + list(ang))
        vals = np.stack(list(ang.values()), axis=1)
        for f in range(d3.shape[0]):
            w.writerow([f, f'{f / fps:.4f}'] + [('' if np.isnan(v) else f'{v:.2f}') for v in vals[f]])
    log(f'exported {path.relative_to(main)}')


# --------------------------------------------------------------------------------------
# 5. QC
# --------------------------------------------------------------------------------------

def runs(mask):
    """Start/end (inclusive) of True runs in a 1D bool array."""
    m = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def shade(ax, mask, t, color='tab:red', alpha=0.15, label=None):
    dt = t[1] - t[0] if len(t) > 1 else 1
    for i, (a, b) in enumerate(runs(mask)):
        ax.axvspan(t[a] - dt / 2, t[b] + dt / 2, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)


def robust_spikes(v, k=6.0):
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    thr = med + k * max(mad, 1e-9)
    return v > thr, thr


def acquisition_qc(rec_dir, main, plt, outdir, summary):
    report = main / 'calibration_report.json'
    if rec_dir is None or not report.exists():
        return
    order = json.loads(report.read_text())['athena_order']
    ts, names = [], []
    for c in order:
        f = rec_dir / f"Metadata_{Path(c['source_video']).stem.replace('Take_', '')}.json"
        if not f.exists():
            cand = list(rec_dir.glob(f"Metadata_*-M{c['serial']}.json"))
            f = cand[0] if cand else None
        if f is None:
            return
        fr = json.loads(f.read_text())['frames']
        ts.append(np.array([x['camera_timestamp'] for x in fr], dtype=np.int64))
        names.append(f"{c['camera']}")
        summary.setdefault('acquisition', {})[c['camera']] = {
            'frames': len(fr), 'camera_skips': int(sum(x.get('estimated_camera_skips_delta', 0) for x in fr))}
    n = min(len(t) for t in ts)
    T = np.stack([t[:n] for t in ts])
    dt = np.diff(T, axis=1) / 1e6
    nominal = np.median(dt)
    skew = (T.max(0) - T.min(0)) / 1e6
    gaps = dt > 1.5 * nominal
    summary['acquisition_overall'] = {'nominal_interval_ms': float(nominal), 'max_interval_ms': float(dt.max()),
                                      'dropped_frame_events': int(gaps.sum()), 'max_skew_ms': float(skew.max())}

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, nm in enumerate(names):
        ax[0].plot(np.arange(1, n), dt[i], lw=0.8, label=nm)
    ax[0].set_ylabel('frame interval [ms]')
    ax[0].set_title(f'Acquisition: nominal {nominal:.3f} ms, max {dt.max():.3f} ms, '
                    f'{int(gaps.sum())} intervals > 1.5x nominal (dropped frames)')
    ax[0].legend(ncol=9, fontsize=7)
    ax[1].imshow(gaps, aspect='auto', interpolation='nearest', cmap='Reds', vmin=0, vmax=1,
                 extent=[0.5, n - 0.5, len(names) - 0.5, -0.5])
    ax[1].set_yticks(range(len(names)), names)
    ax[1].set_title('dropped-frame events per camera (red)')
    ax[2].plot(skew * 1e3, lw=0.8)
    ax[2].set_ylabel('cross-camera skew [µs]')
    ax[2].set_xlabel('frame')
    fig.tight_layout()
    fig.savefig(outdir / '01_acquisition.png', dpi=110)
    plt.close(fig)


QUALITY_STATUS = {0: 'missing', 1: 'interpolated', 2: 'triangulated'}
QUALITY_CLASS = {0: 'missing', 1: 'interpolated', 2: 'suspect', 3: 'reliable'}
# default thresholds of the "reliable" mask (the notebook lets you change them)
QUALITY_DEFAULTS = {'min_cams': 3, 'max_reproj_px': 20.0, 'max_dev_mm': 10.0, 'exclude_near_gap': True}


def point_status(d3, raw):
    """Per frame and landmark: 0 = missing in the final 3D, 1 = interpolated by ATHENA smoothing (no raw
    triangulation: < 2 cameras), 2 = triangulated from >= 2 cameras (then smoothed)."""
    final_ok = np.isfinite(d3).all(-1)
    raw_ok = np.isfinite(raw).all(-1)
    return np.where(final_ok, np.where(raw_ok, 2, 1), 0).astype(np.int8)


def smoothing_window(fps, lfc):
    """Savitzky-Golay window [frames] used by ATHENA smooth3d for the given frame rate and cutoff."""
    return max(3, int(fps / lfc * 2 + 1) | 1)


def reprojection_error(main, raw, sw):
    """Per frame and landmark: mean and max over the cameras that detected the point of the distance [px]
    between the 2D detection and the raw 3D point projected back into that camera."""
    import warnings
    ext, K, _ = load_calibration(main)
    ncams, nframes, nlm, _ = sw.shape
    X = np.concatenate([raw.reshape(-1, 3), np.ones((nframes * nlm, 1))], axis=1)
    err = np.full((ncams, nframes, nlm), np.nan)
    for c in range(ncams):
        x = X @ (np.asarray(K[c]) @ np.asarray(ext[c])[:3, :]).T
        uv = (x[:, :2] / x[:, 2:3]).reshape(nframes, nlm, 2)
        err[c] = np.linalg.norm(uv - sw[c], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(err, 0), np.nanmax(err, 0)


def near_gap_mask(raw, half_window):
    """True for triangulated frames within `half_window` frames of a frame where the raw 3D is missing: their
    smoothing window includes ATHENA's linear fill across the gap, so they are partly distorted."""
    miss = ~np.isfinite(raw).all(-1)
    near = np.zeros_like(miss)
    for k in range(1, half_window + 1):
        near[k:] |= miss[:-k]
        near[:-k] |= miss[k:]
    return near & ~miss


def quality_classes(q, min_cams=None, max_reproj_px=None, max_dev_mm=None, exclude_near_gap=None):
    """Classify every point (frame, landmark): 0 missing, 1 interpolated, 2 suspect, 3 reliable.
    `q`: dict/npz from <trial>_3Dquality.npz. A triangulated point is *suspect* if any check fails:
        few_cams   seen by fewer than min_cams cameras
        reproj     max reprojection error over cameras > max_reproj_px  (wrong detection in some camera)
        outlier    |raw - final| > max_dev_mm                          (spike flattened by the smoothing)
        near_gap   within half a smoothing window of a gap               (distorted by the gap fill)
    Returns (classes int8 array, {reason: bool array})."""
    d = QUALITY_DEFAULTS
    min_cams = d['min_cams'] if min_cams is None else min_cams
    max_reproj_px = d['max_reproj_px'] if max_reproj_px is None else max_reproj_px
    max_dev_mm = d['max_dev_mm'] if max_dev_mm is None else max_dev_mm
    exclude_near_gap = d['exclude_near_gap'] if exclude_near_gap is None else exclude_near_gap
    status = np.asarray(q['status'])
    tri = status == 2
    reasons = {
        'few_cams': tri & (np.asarray(q['ncams']) < min_cams),
        'reproj': tri & (np.nan_to_num(np.asarray(q['reproj_max_px'], float), nan=0) > max_reproj_px),
        'outlier': tri & (np.nan_to_num(np.asarray(q['dev_mm'], float), nan=0) > max_dev_mm),
        'near_gap': tri & np.asarray(q['near_gap'], bool) if exclude_near_gap else np.zeros_like(tri),
    }
    suspect = np.zeros_like(tri)
    for m in reasons.values():
        suspect |= m
    cls = np.where(tri, np.where(suspect, 2, 3), status).astype(np.int8)
    return cls, reasons


def quality_metrics(main, d3, raw, sw, nc, fps, lfc):
    """All per-point quality arrays saved in <trial>_3Dquality.npz."""
    rmean, rmax = reprojection_error(main, raw, sw)
    win = smoothing_window(fps, lfc)
    return {
        'status': point_status(d3, raw), 'ncams': nc.astype(np.uint8), 'raw3d': raw.astype(np.float32),
        'reproj_mean_px': rmean.astype(np.float32), 'reproj_max_px': rmax.astype(np.float32),
        'dev_mm': np.linalg.norm(raw - d3, axis=-1).astype(np.float32),
        'near_gap': near_gap_mask(raw, win // 2), 'smoothing_window': np.int16(win),
    }


def save_quality(outdir, trial, metrics):
    """analysis/<trial>/<trial>_3Dquality.npz (see quality_metrics / quality_classes)."""
    np.savez_compressed(outdir / f'{trial}_3Dquality.npz', **metrics,
                        landmark_names=np.array(LANDMARK_NAMES),
                        legend=np.array([f'{k}={v}' for k, v in QUALITY_STATUS.items()]))


def lfc_used(adir):
    """Smoothing cutoff [Hz] of the last run that computed the 3D (run_options.json or run_options_3d+qc.json)."""
    files = [f for f in (Path(adir) / 'run_options.json', Path(adir) / 'run_options_3d+qc.json') if f.exists()]
    if not files:
        return 12.0
    f = max(files, key=lambda f: f.stat().st_mtime)
    return float(json.loads(f.read_text()).get('all_landmarks_lfc', 12.0))


def compute_quality(main, trial, d3=None, fps=None, lfc=None, save=True):
    """Compute (and save) the quality file of a trial from its 2D landmarks and final 3D."""
    main = Path(main)
    if d3 is None:
        d3 = np.load(main / 'landmarks' / trial / f'{trial}_3Dlandmarks.npy')
    adir = main / 'analysis' / trial
    if lfc is None:
        lfc = lfc_used(adir)
    if fps is None:
        vids = sorted(v for v in (main / 'videos' / trial).glob('*.mp4') if v.exists())
        fps = video_frames(vids[0])[1] if vids else 40.0
    raw, sw, nc = triangulate_raw(main, trial)
    m = quality_metrics(main, d3, raw, sw, nc, fps, lfc)
    if save and adir.is_dir():
        save_quality(adir, trial, m)
    return m


def qc(main, rec_dir, trial, fps, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)

    lmdir = main / 'landmarks' / trial
    d3 = np.load(lmdir / f'{trial}_3Dlandmarks.npy')
    d2ref = np.load(lmdir / f'{trial}_2Dlandmarksrefined.npy')
    R, Lh, B, hs = load_2d(main, trial)
    raw, sw, nc = triangulate_raw(main, trial)
    lfc = lfc_used(outdir)
    qm = quality_metrics(main, d3, raw, sw, nc, fps, lfc)
    save_quality(outdir, trial, qm)
    ncams, nframes = R.shape[:2]
    t = np.arange(nframes) / fps
    camnames = [p.stem for p in sorted((main / 'calibration').glob('*.yaml'))] or [f'cam{i}' for i in range(ncams)]
    summary = {'trial': trial, 'frames': int(nframes), 'fps': fps, 'duration_s': nframes / fps, 'cameras': camnames}

    acquisition_qc(rec_dir, main, plt, outdir, summary)

    # ---- 02 detection timeline
    det = {'right hand': R[..., 0, 0] >= 0, 'left hand': Lh[..., 0, 0] >= 0, 'body': B[..., 0, 0] >= 0}
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for a, (k, m) in zip(ax, det.items()):
        a.imshow(m, aspect='auto', interpolation='nearest', cmap='Greens', vmin=0, vmax=1,
                 extent=[t[0], t[-1], ncams - 0.5, -0.5])
        a.set_yticks(range(ncams), [f'{c} ({100 * m[i].mean():.0f}%)' for i, c in enumerate(camnames)], fontsize=8)
        a.set_title(f'2D detection (MediaPipe label) – {k}: green = detected')
    ax[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(outdir / '02_detection_2d.png', dpi=110)
    plt.close(fig)
    summary['detection_2d_percent'] = {k: {c: float(100 * m[i].mean()) for i, c in enumerate(camnames)}
                                       for k, m in det.items()}

    # ---- 03 coverage + 04 gaps
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig2, ax2 = plt.subplots(2, 2, figsize=(13, 7))
    gap_rows = []
    summary['hands'] = {}
    for i, (hand, o) in enumerate(HANDS.items()):
        ncam_wrist = nc[:, o]
        raw_ok = np.isfinite(raw[:, o:o + 21, 0]).all(1)
        fin_ok = np.isfinite(d3[:, o:o + 21, 0]).all(1)
        filled = fin_ok & ~raw_ok
        a = ax[i]
        a.step(t, ncam_wrist, where='mid', color='k', lw=0.8, label='cameras seeing wrist (after hand switching)')
        a.axhline(2, color='tab:orange', ls='--', lw=0.8, label='min. 2 cameras')
        shade(a, ~fin_ok, t, 'tab:red', 0.25, 'missing in final 3D')
        shade(a, filled, t, 'tab:blue', 0.25, 'interpolated by smoothing (raw missing)')
        a.set_ylim(-0.2, ncams + 0.5)
        a.set_ylabel('# cameras')
        a.set_title(f'{hand} hand: raw 3D valid {100 * raw_ok.mean():.1f}%  |  final 3D valid {100 * fin_ok.mean():.1f}%'
                    f'  |  interpolated {100 * filled.mean():.1f}%')
        a.legend(loc='upper right', fontsize=7, ncol=4)

        g_raw = [(s, e) for s, e in runs(~raw_ok)]
        g_fin = [(s, e) for s, e in runs(~fin_ok)]
        for kind, gl in (('raw', g_raw), ('final', g_fin)):
            for s, e in gl:
                gap_rows.append([hand, kind, s, e, e - s + 1, f'{s / fps:.3f}', f'{(e - s + 1) / fps:.3f}'])
        lens_raw = np.array([e - s + 1 for s, e in g_raw]) / fps
        lens_fin = np.array([e - s + 1 for s, e in g_fin]) / fps
        b = ax2[0, i]
        for s, e in g_raw:
            b.barh(1, (e - s + 1) / fps, left=s / fps, color='tab:orange')
        for s, e in g_fin:
            b.barh(0, (e - s + 1) / fps, left=s / fps, color='tab:red')
        b.set_yticks([0, 1], ['final 3D', 'raw 3D'])
        b.set_xlim(t[0], t[-1] + 1 / fps)
        b.set_xlabel('time [s]')
        b.set_title(f'{hand} hand gaps: raw {len(g_raw)}, final {len(g_fin)}')
        c = ax2[1, i]
        bins = np.logspace(np.log10(1 / fps), np.log10(max(nframes / fps, 2 / fps)), 25)
        if len(lens_raw):
            c.hist(lens_raw, bins=bins, alpha=0.6, color='tab:orange', label='raw')
        if len(lens_fin):
            c.hist(lens_fin, bins=bins, alpha=0.6, color='tab:red', label='final')
        c.set_xscale('log')
        c.set_xlabel('gap duration [s]')
        c.set_ylabel('count')
        if len(lens_raw) or len(lens_fin):
            c.legend(fontsize=8)
        summary['hands'][hand] = {
            'raw_valid_percent': float(100 * raw_ok.mean()), 'final_valid_percent': float(100 * fin_ok.mean()),
            'interpolated_percent': float(100 * filled.mean()),
            'n_gaps_raw': len(g_raw), 'n_gaps_final': len(g_fin),
            'longest_gap_raw_s': float(lens_raw.max()) if len(lens_raw) else 0.0,
            'longest_gap_final_s': float(lens_fin.max()) if len(lens_fin) else 0.0,
            'median_cameras_wrist': float(np.median(ncam_wrist))}
    ax[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(outdir / '03_coverage.png', dpi=110)
    fig2.tight_layout()
    fig2.savefig(outdir / '04_gaps.png', dpi=110)
    plt.close(fig)
    plt.close(fig2)
    with open(outdir / 'gaps.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['hand', 'data', 'start_frame', 'end_frame', 'n_frames', 'start_s', 'duration_s'])
        w.writerows(gap_rows)

    # ---- 05 time series per hand
    for hand, o in HANDS.items():
        fin_ok = np.isfinite(d3[:, o:o + 21, 0]).all(1)
        if not np.isfinite(d3[:, o]).any():
            continue
        fig, ax = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
        spikes_tot = {}
        for r, axis in enumerate('xyz'):
            a = ax[r]
            for kp, j in KEY_POINTS.items():
                ln, = a.plot(t, d3[:, o + j, r], lw=1, label=f'{kp} (final)')
                a.plot(t, raw[:, o + j, r], '.', ms=2, color=ln.get_color(), alpha=0.5)
            shade(a, ~fin_ok, t)
            a.set_ylabel(f'{axis} [mm]')
        ax[0].legend(fontsize=7, ncol=3, loc='upper right')
        ax[0].set_title(f'{hand} hand – lines: final (smoothed) 3D, dots: raw triangulation, red: missing')
        a = ax[3]
        for kp, j in KEY_POINTS.items():
            v = np.linalg.norm(np.diff(d3[:, o + j], axis=0), axis=1) * fps
            sp, thr = robust_spikes(v)
            ln, = a.plot(t[1:], v, lw=0.8, label=kp)
            a.plot(t[1:][sp], v[sp], 'x', color=ln.get_color(), ms=6)
            spikes_tot[kp] = int(np.nansum(sp))
        a.set_ylabel('speed [mm/s]')
        a.set_title('speed of final 3D (x = spikes > median + 6·MAD)')
        a.legend(fontsize=7)
        a = ax[4]
        dist_f = np.linalg.norm(d3[:, o + 4] - d3[:, o + 8], axis=1)
        dist_r = np.linalg.norm(raw[:, o + 4] - raw[:, o + 8], axis=1)
        a.plot(t, dist_f, 'k', lw=1, label='final')
        a.plot(t, dist_r, '.', ms=2, color='tab:gray', label='raw')
        shade(a, ~fin_ok, t)
        a.set_ylabel('thumb-index [mm]')
        a.set_title('thumb tip – index tip distance (tapping signal)')
        a.legend(fontsize=7)
        a = ax[5]
        a.step(t, nc[:, o + 8], where='mid', color='k', lw=0.8, label='index tip')
        a.step(t, nc[:, o + 4], where='mid', color='tab:purple', lw=0.8, label='thumb tip')
        a.set_ylabel('# cameras')
        a.set_xlabel('time [s]')
        a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f'05_timeseries_{hand}.png', dpi=100)
        plt.close(fig)
        summary['hands'][hand]['speed_spikes'] = spikes_tot
        summary['hands'][hand]['raw_vs_final_rmse_mm'] = float(
            np.sqrt(np.nanmean(np.sum((raw[:, o:o + 21] - d3[:, o:o + 21]) ** 2, axis=-1))))

    # ---- point reliability with the default thresholds (QUALITY_DEFAULTS; adjustable in the notebook)
    cls, reasons = quality_classes(qm)
    summary['quality_thresholds'] = dict(QUALITY_DEFAULTS, smoothing_window_frames=int(qm['smoothing_window']))
    for hand, o in HANDS.items():
        c = cls[:, o:o + 21]
        summary['hands'][hand].update({
            'reliable_percent': float(100 * (c == 3).mean()), 'suspect_percent': float(100 * (c == 2).mean()),
            'interpolated_percent': float(100 * (c == 1).mean()), 'missing_percent': float(100 * (c == 0).mean()),
            'suspect_reasons_percent': {k: float(100 * v[:, o:o + 21].mean()) for k, v in reasons.items()},
            'reproj_max_px_median': float(np.nanmedian(qm['reproj_max_px'][:, o:o + 21])),
            'raw_final_dev_mm_p99': float(np.nanpercentile(qm['dev_mm'][:, o:o + 21], 99)),
        })

    # ---- 06 reprojection error (2D detections after hand switching vs reprojected final 3D)
    err = np.linalg.norm(sw - d2ref, axis=-1)  # ncams, nframes, nlm
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    summary['reprojection_px'] = {}
    for i, (hand, o) in enumerate(HANDS.items()):
        e = err[:, :, o:o + 21]
        data = [e[c][np.isfinite(e[c])] for c in range(ncams)]
        a = ax[0, i]
        a.boxplot([x if len(x) else [np.nan] for x in data], showfliers=False)
        a.set_xticks(range(1, ncams + 1), camnames, rotation=45, fontsize=8)
        a.set_ylabel('error [px]')
        a.set_title(f'{hand} hand reprojection error per camera (n = detections)')
        for c, x in enumerate(data):
            a.text(c + 1, 0, f'{len(x)}', ha='center', va='bottom', fontsize=6, color='gray')
        a = ax[1, i]
        a.plot(t, np.nanmedian(e, axis=(0, 2)), lw=0.8, color='k', label='median over cams/landmarks')
        a.plot(t, np.nanmax(np.nanmedian(e, axis=2), axis=0), lw=0.6, color='tab:red', label='worst camera')
        a.set_yscale('log')
        a.set_xlabel('time [s]')
        a.set_ylabel('error [px]')
        a.legend(fontsize=7)
        summary['reprojection_px'][hand] = {camnames[c]: (float(np.median(x)) if len(x) else None)
                                            for c, x in enumerate(data)}
    fig.tight_layout()
    fig.savefig(outdir / '06_reprojection.png', dpi=110)
    plt.close(fig)

    # ---- 07 bone lengths
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    summary['bone_length_cv_percent'] = {}
    for i, (hand, o) in enumerate(HANDS.items()):
        L = np.stack([np.linalg.norm(d3[:, o + a] - d3[:, o + b], axis=1) for a, b in HAND_BONES], axis=1)
        data = [x[np.isfinite(x)] for x in L.T]
        ax[i].boxplot([x if len(x) else [np.nan] for x in data], showfliers=True, flierprops={'markersize': 2})
        ax[i].set_xticks(range(1, len(HAND_BONES) + 1),
                         [f'{HAND_NAMES[b]}' for a, b in HAND_BONES], rotation=70, fontsize=7)
        ax[i].set_ylabel('length [mm]')
        cv = {HAND_NAMES[b]: (float(100 * np.std(x) / np.mean(x)) if len(x) else None)
              for (a, b), x in zip(HAND_BONES, data)}
        med = np.nanmedian([v for v in cv.values() if v is not None]) if any(cv.values()) else float('nan')
        ax[i].set_title(f'{hand} hand bone lengths (median CV {med:.1f}%)  – should be constant')
        summary['bone_length_cv_percent'][hand] = cv
    fig.tight_layout()
    fig.savefig(outdir / '07_bone_lengths.png', dpi=110)
    plt.close(fig)

    (outdir / 'qc_summary.json').write_text(json.dumps(summary, indent=2))
    return d3, summary


def print_summary(s):
    print('\n================ QC SUMMARY ================')
    print(f"{s['trial']}: {s['frames']} frames, {s['duration_s']:.1f} s @ {s['fps']:g} fps")
    if 'acquisition_overall' in s:
        a = s['acquisition_overall']
        print(f"acquisition: dropped-frame events {a['dropped_frame_events']}, max interval "
              f"{a['max_interval_ms']:.2f} ms, max skew {a['max_skew_ms'] * 1e3:.3f} us")
    for h, d in s['hands'].items():
        print(f"{h:5s} hand: raw 3D {d['raw_valid_percent']:5.1f}%  final {d['final_valid_percent']:5.1f}%  "
              f"gaps raw/final {d['n_gaps_raw']}/{d['n_gaps_final']}  longest final gap {d['longest_gap_final_s']:.2f} s")
        if 'reliable_percent' in d:
            r = d['suspect_reasons_percent']
            print(f"           reliable {d['reliable_percent']:5.1f}%  suspect {d['suspect_percent']:5.1f}% "
                  f"(<{s['quality_thresholds']['min_cams']} cams {r['few_cams']:.1f}%, reproj {r['reproj']:.1f}%, "
                  f"outlier {r['outlier']:.1f}%, near gap {r['near_gap']:.1f}%)")
    print('=============================================\n')


# --------------------------------------------------------------------------------------
# Run log (execution time + info to estimate run time of future recordings)
# --------------------------------------------------------------------------------------

DEFAULT_RUN_LOG = Path(__file__).resolve().parent.parent / 'athena_runs_log.csv'

# Output profiles: landmarks + analysis/QC + videos_processed/<trial>/data3d.mp4 are always produced
OUTPUT_PROFILES = {
    'minimal':  {'overlays': False, 'keep_frames': False},  # no 2D / refined overlay videos
    'standard': {'overlays': True,  'keep_frames': False},  # overlay videos, per-frame images deleted
    'full':     {'overlays': True,  'keep_frames': True},   # everything ATHENA can write
}


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def hardware_info():
    info = {'hostname': os.uname().nodename, 'cpu_count': os.cpu_count()}
    try:
        info['cpu_model'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        info['cpu_model'] = ''
    try:
        kb = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemTotal'))
        info['ram_gb'] = round(kb / 1024 ** 2, 1)
    except Exception:
        info['ram_gb'] = ''
    gpu = _cmd(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    info['gpu'] = gpu.splitlines()[0] if gpu else ''
    try:
        import mediapipe
        info['mediapipe_version'] = mediapipe.__version__
    except Exception:
        info['mediapipe_version'] = ''
    try:
        from importlib.metadata import version
        info['athena_version'] = version('athena-tracking')
    except Exception:
        info['athena_version'] = ''
    return info


def dir_mb(*paths):
    tot = 0
    for p in paths:
        if p.is_dir():
            tot += sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    return round(tot / 1e6, 1)


def count_files(p, pattern='*'):
    return sum(1 for _ in p.rglob(pattern)) if p.is_dir() else 0


def append_run_log(path, row):
    """Append a row; if new columns appear, rewrite the file with the union of the columns."""
    path = Path(path)
    rows, header = [], list(row.keys())
    if path.exists():
        with open(path, newline='') as fh:
            r = csv.DictReader(fh)
            rows = list(r)
            old = r.fieldnames or []
        header = old + [k for k in row if k not in old]
    rows.append({k: row.get(k, '') for k in header})
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})


# --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?', type=Path, help='recording folder or ATHENA main folder (dialog if omitted)')
    ap.add_argument('--trial', action='append', help='trial (videos/<trial>) to process; repeatable. Default: all')
    ap.add_argument('--fraction', type=float, default=1.0, help='fraction of frames to process (test: 0.1)')
    ap.add_argument('--processes', type=int, default=min(os.cpu_count() or 1, 9))
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--hand-conf', type=float, default=0.9, help='MediaPipe hand confidence (GUI default 0.9)')
    ap.add_argument('--pose-conf', type=float, default=0.9, help='MediaPipe pose confidence (GUI default 0.9)')
    ap.add_argument('--lfc', type=float, default=12.0, help='smoothing low-pass cutoff [Hz]')
    o = ap.add_argument_group('outputs / storage')
    o.add_argument('--outputs', choices=OUTPUT_PROFILES, default='standard',
                   help='minimal: data + QC + data3d.mp4 | standard (default): + 2D and refined overlay videos | '
                        'full: as standard and keep the per-frame images')
    o.add_argument('--keep-frames', action='store_true', help='keep per-frame images (images/, imagesrefined/)')
    o.add_argument('--light', action='store_true', help='(old option) same as --outputs minimal')
    o.add_argument('--video-link', choices=('symlink', 'hardlink', 'copy'), default='symlink',
                   help='how athena/videos/<trial>/cam_N.mp4 refers to Take_*.mkv (default: relative symlink, 0 bytes)')
    o.add_argument('--cleanup', action='store_true',
                   help='do not analyse: shrink recordings already processed to the --outputs profile '
                        '(video copies -> links, delete per-frame images and videos not in the profile)')
    o.add_argument('--dry-run', action='store_true', help='with --cleanup: only show what would be removed')
    ap.add_argument('--skip-2d', action='store_true', help='reuse existing 2D landmarks')
    ap.add_argument('--skip-3d', action='store_true', help='reuse existing 3D landmarks (only export + QC)')
    ap.add_argument('--run-log', type=Path, default=DEFAULT_RUN_LOG,
                    help=f'CSV where every run is appended (default: {DEFAULT_RUN_LOG})')
    g = ap.add_argument_group('calibration / batch')
    g.add_argument('--square-mm', type=float, default=100.0,
                   help='checkerboard square [mm] used to calibrate recordings without calibration '
                        '(athena_calibration.py). Default: 100 (lab board, 10 cm squares)')
    g.add_argument('--recalibrate', action='store_true', help='recompute calibration even if present')
    g.add_argument('--calib-args', default='', help='extra arguments for athena_calibration.py, e.g. "--origin center"')
    g.add_argument('--redo', action='store_true', help='batch: re-process recordings already completed')
    g.add_argument('--skip-done', action='store_true', help='single recording: skip trials already completed')
    g.add_argument('--select', action='append', metavar='PATTERN',
                   help='batch: process only recordings whose name matches (repeatable; text contained in the name, '
                        'or wildcards: "*roberto*", "2026-11-10_11*")')
    g.add_argument('--exclude', action='append', metavar='PATTERN', help='batch: skip recordings matching (repeatable)')
    g.add_argument('--list', action='store_true',
                   help='batch: only list the recordings (after --select/--exclude) with their status, then exit')
    return ap.parse_args()


def main():
    args = parse_args()
    folder = args.folder
    if folder is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        sel = filedialog.askdirectory(title='Select recording folder or ATHENA main folder')
        root.destroy()
        if not sel:
            sys.exit('no folder selected')
        folder = Path(sel)
    folder = folder.resolve()

    if args.light:
        args.outputs = 'minimal'
    overlays = OUTPUT_PROFILES[args.outputs]['overlays']
    opts = {
        'fraction_frames': args.fraction, 'num_processes': args.processes, 'use_gpu': not args.no_gpu,
        'hand_confidence': args.hand_conf, 'pose_confidence': args.pose_conf,
        'save_images_mp': overlays, 'save_video_mp': overlays,
        'all_landmarks_lfc': args.lfc,
        'save_images_triangulation': True, 'save_video_triangulation': True,  # data3d.mp4: always
        'save_images_refine': overlays, 'save_video_refine': overlays,
        # not ATHENA options (ignored by ATHENA): output profile
        'outputs_profile': args.outputs,
        'keep_frames': args.keep_frames or OUTPUT_PROFILES[args.outputs]['keep_frames'],
        'video_link': args.video_link,
    }

    recordings = find_recordings(folder)
    if recordings and (args.select or args.exclude):
        recordings = [r for r in recordings if (not args.select or any(name_match(r.name, p) for p in args.select))
                      and not any(name_match(r.name, p) for p in (args.exclude or []))]
        if not recordings:
            sys.exit(f'no recording in {folder} matches --select {args.select} / --exclude {args.exclude}')
    if args.list:
        list_recordings(recordings or [folder], opts)
        return
    if args.cleanup:
        tot = 0
        for rec in (recordings or [folder]):
            try:
                tot += cleanup_recording(rec, opts, dry_run=args.dry_run)
            except SystemExit as e:
                log(f'{Path(rec).name}: skipped ({e})')
        log(f"{'would free' if args.dry_run else 'freed'} {tot / 1e9:.2f} GB in total"
            + (' (dry run: nothing deleted)' if args.dry_run else ''))
        return

    hw = hardware_info()

    if not recordings:  # single recording / main folder
        process_recording(folder, args, opts, hw, skip_done=args.skip_done)
        return

    # ---------------- batch mode: folder containing several recordings ----------------
    log(f'BATCH mode: {len(recordings)} recordings in {folder}')
    for r in recordings:
        log(f'  - {r.name}')
    results = []
    t_batch = time.time()
    for i, rec in enumerate(recordings, 1):
        print(f'\n{"#" * 90}\n# [{i}/{len(recordings)}] {rec.name}\n{"#" * 90}', flush=True)
        t0 = time.time()
        try:
            status = process_recording(rec, args, opts, hw, skip_done=not args.redo)
            results.append((rec.name, status, time.time() - t0, ''))
        except KeyboardInterrupt:
            results.append((rec.name, 'interrupted', time.time() - t0, ''))
            print_batch_summary(results, t_batch)
            raise
        except BaseException as e:  # SystemExit from checks or any error: log it and go on
            msg = str(e) if isinstance(e, SystemExit) else f'{type(e).__name__}: {e}'
            log(f'FAILED {rec.name}: {msg}')
            results.append((rec.name, 'error', time.time() - t0, msg))
            if not isinstance(e, SystemExit):
                import traceback
                traceback.print_exc()
            if not getattr(e, 'run_logged', False):  # failed before any trial started (calibration, folders, ...)
                append_run_log(args.run_log, {'run_start': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': 'error',
                                              'error': msg[:300], 'recording': rec.name,
                                              't_total_s': round(time.time() - t0, 2), **hw})
    print_batch_summary(results, t_batch)


def print_batch_summary(results, t_batch):
    print(f'\n{"=" * 30} BATCH SUMMARY {"=" * 30}')
    for name, status, dt, msg in results:
        print(f'{status:12s} {dt / 60:7.1f} min  {name}' + (f'   <- {msg[:120]}' if msg else ''))
    n_ok = sum(r[1] in ('ok', 'skipped') for r in results)
    print(f'{n_ok}/{len(results)} ok/skipped, total {(time.time() - t_batch) / 60:.1f} min')
    print('=' * 75)


def name_match(name, pattern):
    """Wildcard match if the pattern has * ? [ ], otherwise case-insensitive substring."""
    import fnmatch
    if any(c in pattern for c in '*?['):
        return fnmatch.fnmatch(name, pattern)
    return pattern.lower() in name.lower()


def list_recordings(recordings, opts):
    """Print one line per recording: status, calibration, number/size of raw videos, duration."""
    print(f'{"status":10s} {"calib":5s} {"videos":>6s} {"GB":>6s} {"dur_s":>6s}  recording')
    n_todo = 0
    for rec in recordings:
        rec = Path(rec)
        main = rec if (rec / 'calibration').is_dir() else rec / 'athena'
        takes = sorted(rec.glob('Take_*'))
        calib = has_calibration(main)
        dur = ''
        try:
            trials = [d.name for d in (main / 'videos').iterdir() if d.is_dir()] if (main / 'videos').is_dir() else []
            trials = trials or [DATE_PREFIX.sub('', rec.name) or rec.name]
            done = all(trial_done(main, t, opts) for t in trials)
            info = json.loads((main / 'analysis' / trials[0] / 'run_info.json').read_text()) \
                if (main / 'analysis' / trials[0] / 'run_info.json').exists() else {}
            dur = info.get('video_duration_s', '')
        except Exception:
            done = False
        if not dur and takes:
            try:
                n, fps = video_frames(takes[0])
                dur = round(n / fps, 1)
            except Exception:
                pass
        n_todo += not done
        gb = sum(t.stat().st_size for t in takes) / 1e9
        print(f'{"done" if done else "to do":10s} {"yes" if calib else "no":5s} {len(takes):6d} {gb:6.1f} '
              f'{dur if dur != "" else "?":>6}  {rec.name}')
    print(f'{len(recordings)} recordings, {n_todo} to do')


def is_recording(p):
    p = Path(p)
    return p.is_dir() and (any(p.glob('Take_*')) or (p / 'calibration_frames').is_dir()
                           or (p / 'athena' / 'calibration').is_dir())


def find_recordings(folder):
    """Recordings directly inside `folder` (batch mode). Empty if `folder` itself is a recording/main folder."""
    folder = Path(folder)
    if is_recording(folder) or (folder / 'calibration').is_dir():
        return []
    return sorted(d for d in folder.iterdir() if is_recording(d))


def has_calibration(main):
    c = Path(main) / 'calibration'
    return c.is_dir() and (any(c.glob('*.yaml')) or any(c.glob('*.toml')))


def ensure_calibration(folder, args):
    """Run athena_calibration.py on a recording if it has no calibration (or --recalibrate)."""
    folder = Path(folder)
    if (folder / 'calibration').is_dir():  # already an ATHENA main folder
        return
    main = folder / 'athena'
    if has_calibration(main) and not args.recalibrate:
        return
    if args.square_mm is None:
        sys.exit(f'no calibration in {main}/calibration: pass --square-mm to compute it automatically '
                 f'(or run athena_calibration.py)')
    if not (folder / 'calibration_frames').is_dir():
        sys.exit(f'no calibration and no calibration_frames/ in {folder}')
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'athena_calibration.py'), str(folder),
           '--square-mm', str(args.square_mm)] + (args.calib_args.split() if args.calib_args else [])
    log('calibration: ' + ' '.join(cmd[1:]))
    subprocess.run(cmd, check=True)


def trial_done(main_dir, trial, opts):
    f = Path(main_dir) / 'analysis' / trial / 'run_info.json'
    if not f.exists():
        return False
    try:
        r = json.loads(f.read_text())
    except Exception:
        return False
    return (r.get('status') == 'ok' and r.get('steps') == '2d+3d+qc'
            and float(r.get('fraction', 0)) >= opts['fraction_frames'])


def cleanup_recording(folder, opts, dry_run=False):
    """Shrink an already processed recording to the output profile in `opts`, without re-analysing it.
    Returns the number of bytes freed (or that would be freed with dry_run)."""
    main, rec_dir = resolve_folders(Path(folder))
    overlays = OUTPUT_PROFILES[opts['outputs_profile']]['overlays']
    keep, mode = opts['keep_frames'], opts['video_link']
    tag = '[dry-run] ' if dry_run else ''
    freed = 0
    log(f'===== cleanup {(rec_dir or main).name}  (profile {opts["outputs_profile"]}, '
        f'keep frames {keep}, videos {mode}) =====')

    def remove(path, what):
        nonlocal freed
        size = dir_mb(path) * 1e6 if path.is_dir() else path.stat().st_size
        log(f'{tag}delete {what}: {path.relative_to(main)}  ({size / 1e6:.0f} MB)')
        freed += size
        if not dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    trials = sorted({d.name for sub in ('videos', 'images', 'imagesrefined', 'videos_processed')
                     if (main / sub).is_dir() for d in (main / sub).iterdir() if d.is_dir()})

    # 1. athena/videos: full copies of Take_*.mkv -> links
    if mode != 'copy' and rec_dir is not None and (main / 'calibration_report.json').exists():
        mapping = video_mapping(main, rec_dir)
        for trial in trials:
            for src, name in mapping:
                dst = main / 'videos' / trial / name
                if not dst.exists() or not src.exists() or is_linked(src, dst):
                    continue
                if video_frames(src)[0] != video_frames(dst)[0]:
                    log(f'WARNING: {dst.relative_to(main)} and {src.name} differ in frame count: left untouched')
                    continue
                size = dst.stat().st_size
                log(f'{tag}video copy -> {mode}: {dst.relative_to(main)} -> {src.name}  ({size / 1e6:.0f} MB)')
                freed += size
                if not dry_run:
                    link_video(src, dst, mode)

    # 2. per-frame images and overlay videos
    for trial in trials:
        vp = main / 'videos_processed' / trial
        vids = sorted(v for v in (main / 'videos' / trial).glob('*.mp4') if v.exists())
        fps = video_frames(vids[0])[1] if vids else 30.0
        groups = []  # (frames folder, extension, video, wanted by the profile)
        for d in sorted((main / 'images' / trial).glob('cam*')):
            groups.append((d, '.png', vp / f'{d.name}.mp4', overlays))
        for d in sorted((main / 'imagesrefined' / trial).glob('cam*')):
            groups.append((d, '.jpg', vp / f'{d.name}_refined.mp4', overlays))
        d3 = main / 'imagesrefined' / trial / 'data3d'
        if d3.is_dir():
            groups.append((d3, '.jpg', vp / 'data3d.mp4', True))
        for frames, ext, video, wanted in groups:
            if wanted and not video.exists() and not keep:
                log(f'{tag}create missing {video.relative_to(main)} from {frames.relative_to(main)}')
                if not dry_run:
                    from athena.labels2d import createvideo
                    createvideo(str(frames), ext, fps, str(vp), video.name)
                    if not video.exists():
                        log(f'WARNING: could not create {video.name}: frames kept')
                        continue
            if not keep:
                remove(frames, 'per-frame images')
        if not overlays and vp.is_dir():
            for v in sorted(vp.glob('cam*.mp4')):
                remove(v, 'overlay video (not in profile)')
        for sub in ('images', 'imagesrefined'):
            p = main / sub / trial
            if not dry_run and p.is_dir() and not any(p.rglob('*')):
                p.rmdir()
        if not (vp / 'data3d.mp4').exists() and not d3.is_dir():
            log(f'WARNING: {trial}: data3d.mp4 missing -> rerun with --skip-2d to create it')
    log(f'{tag}{(rec_dir or main).name}: {freed / 1e9:.2f} GB')
    return freed


def process_recording(folder, args, opts, hw, skip_done=False):
    """Calibration (if needed) + full pipeline on every trial of one recording. Returns 'ok' or 'skipped'."""
    ensure_calibration(folder, args)
    main_dir, rec_dir = resolve_folders(Path(folder))
    trials = args.trial or default_trials(main_dir, rec_dir)
    log(f'main folder: {main_dir}')
    log(f'trials: {trials}')
    n_skipped = 0
    for trial in trials:
        if skip_done and trial_done(main_dir, trial, opts):
            log(f'trial {trial} already completed (analysis/{trial}/run_info.json) -> skipped. Use --redo to force.')
            n_skipped += 1
            continue
        log(f'===== trial {trial} =====')
        row = {'run_start': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': 'running', 'error': '',
               'recording': rec_dir.name if rec_dir else main_dir.name, 'trial': trial, 'main_folder': str(main_dir),
               'steps': '+'.join(s for s, on in (('2d', not (args.skip_2d or args.skip_3d)), ('3d', not args.skip_3d),
                                                 ('qc', True)) if on)}
        times, n_img = {}, {}
        t_all = time.time()
        try:
            t0 = time.time()
            _, fps, vinfo = prepare_videos(main_dir, rec_dir, trial, opts['video_link'])
            times['t_prepare_s'] = time.time() - t0
            outdir = main_dir / 'analysis' / trial
            outdir.mkdir(parents=True, exist_ok=True)
            # run_options.json = options of the last full run; partial runs (--skip-2d/--skip-3d) get their own file
            ro_name = 'run_options.json' if row['steps'] == '2d+3d+qc' else f"run_options_{row['steps']}.json"
            (outdir / ro_name).write_text(json.dumps(opts, indent=2))
            if not args.skip_2d and not args.skip_3d:
                log('ATHENA 2D detection ...')
                t0 = time.time()
                n_img['2d'] = run_athena_2d(main_dir, trial, opts)
                times['t_2d_s'] = time.time() - t0
            if not args.skip_3d:
                log('ATHENA triangulation / refinement ...')
                t0 = time.time()
                n_img['3d'] = run_athena_3d(main_dir, trial, opts)
                times['t_3d_s'] = time.time() - t0
            log('QC analysis ...')
            t0 = time.time()
            d3, summary = qc(main_dir, rec_dir, trial, fps, outdir)
            export(main_dir, trial, d3, fps, outdir)
            times['t_qc_export_s'] = time.time() - t0
            row['status'] = 'ok'
        except BaseException as e:  # also SystemExit / KeyboardInterrupt: log the failed run, then re-raise
            row['status'] = 'interrupted' if isinstance(e, KeyboardInterrupt) else 'error'
            row['error'] = f'{type(e).__name__}: {e}'[:300]
            e.run_logged = True  # the finally below writes the row: batch mode must not log it again
            raise
        finally:
            times['t_total_s'] = time.time() - t_all
            vinfo = locals().get('vinfo', {})
            s = locals().get('summary', {})
            nproc = s.get('frames') or int(opts['fraction_frames'] * vinfo.get('n_frames', 0))  # real frames if known
            ncam = vinfo.get('n_cameras', 0)
            hands = s.get('hands', {})
            row.update({
                # video
                'n_cameras': ncam, 'width': vinfo.get('width', ''), 'height': vinfo.get('height', ''),
                'fps': vinfo.get('fps', ''), 'codec': vinfo.get('codec', ''),
                'bitrate_mbps_per_cam': vinfo.get('bitrate_mbps', ''), 'videos_total_mb': vinfo.get('videos_total_mb', ''),
                'video_frames': vinfo.get('n_frames', ''), 'video_duration_s': vinfo.get('video_duration_s', ''),
                # processing options
                'fraction': opts['fraction_frames'], 'processed_frames': nproc,
                'processed_duration_s': round(nproc / vinfo['fps'], 3) if vinfo.get('fps') else '',
                'processed_camera_frames': nproc * ncam,
                'use_gpu': opts['use_gpu'], 'processes': opts['num_processes'],
                'hand_conf': opts['hand_confidence'], 'pose_conf': opts['pose_confidence'], 'lfc_hz': opts['all_landmarks_lfc'],
                'save_2d_images_video': opts['save_images_mp'],
                'save_3d_images_video': opts['save_images_triangulation'],
                'save_refined_images_video': opts['save_images_refine'],
                'outputs_profile': opts['outputs_profile'], 'keep_frames': opts['keep_frames'],
                'video_link': opts['video_link'],
                # timings
                **{k: round(v, 2) for k, v in times.items()},
                # normalized throughput (for estimating future runs)
                's_per_camera_frame_2d': round(times['t_2d_s'] / (nproc * ncam), 5) if 't_2d_s' in times and nproc * ncam else '',
                'fps_per_camera_2d': round(nproc / times['t_2d_s'], 2) if 't_2d_s' in times and nproc else '',
                's_per_frame_3d': round(times['t_3d_s'] / nproc, 4) if 't_3d_s' in times and nproc else '',
                'realtime_factor_total': round(times['t_total_s'] / (nproc / vinfo['fps']), 2) if nproc and vinfo.get('fps') else '',
                # extracted data
                'right_hand_valid_pct': round(hands.get('right', {}).get('final_valid_percent', float('nan')), 1) if hands else '',
                'left_hand_valid_pct': round(hands.get('left', {}).get('final_valid_percent', float('nan')), 1) if hands else '',
                'right_hand_gaps': hands.get('right', {}).get('n_gaps_final', '') if hands else '',
                'left_hand_gaps': hands.get('left', {}).get('n_gaps_final', '') if hands else '',
                'n_images_written': sum(n_img.values()),  # per-frame images rendered (also if deleted after)
                'n_videos_written': count_files(main_dir / 'videos_processed' / trial, '*.mp4'),
                'data3d_video': (main_dir / 'videos_processed' / trial / 'data3d.mp4').exists(),
                'output_landmarks_mb': dir_mb(main_dir / 'landmarks' / trial),
                'output_images_videos_mb': dir_mb(main_dir / 'images' / trial, main_dir / 'imagesrefined' / trial,
                                                  main_dir / 'videos_processed' / trial),
                'output_analysis_mb': dir_mb(main_dir / 'analysis' / trial),
                # machine
                **hw,
            })
            try:
                append_run_log(args.run_log, row)
                if (main_dir / 'analysis' / trial).is_dir():
                    # run_info.json = last full run (used to skip completed trials); partial runs get their own file
                    name = 'run_info.json' if row['steps'] == '2d+3d+qc' else f"run_info_{row['steps']}.json"
                    (main_dir / 'analysis' / trial / name).write_text(json.dumps(row, indent=2, default=str))
                log(f"run time {times['t_total_s'] / 60:.1f} min -> logged in {args.run_log}")
            except Exception as e:
                log(f'WARNING: could not write run log: {e}')
        print_summary(summary)
        log(f'QC plots in {outdir}')
    return 'skipped' if n_skipped == len(trials) else 'ok'


if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method('spawn', force=True)
    main()
