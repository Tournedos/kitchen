#!/usr/bin/env python3
"""
Generate an ATHENA-compatible multi-camera calibration from the synchronized
checkerboard screenshots saved by the multicam_capture acquisition system.

Input  (one recording folder, e.g. data/2026-11-10_112048_louis_finertapping_01):
    calibration_frames/<capture>/cam_<N>_frame<K>.png   synchronized board screenshots
    calibration_frames/<capture>/camera_config.json     cam name -> serial, lens focal length
    Take_<stamp>-M<serial>.mkv                          videos (used only for checks / mapping)

Output (an ATHENA "main folder"):
    <out>/calibration/cam_<N>.yaml        one file per camera, JARVIS/OpenCV format (default)
      or <out>/calibration/calibration.toml  Anipose format (--format toml)
    <out>/calibration_report.json         per-camera details, errors, camera <-> video mapping
    <out>/calibration_check/cam_<N>.jpg   overlays (corners, world axes) to visually verify

Conventions (same as ATHENA / JARVIS):
    * units: millimetres
    * x_cam = R @ X_world + T ;  pixel ~ K @ x_cam  (+ OpenCV distortion)
    * the YAML stores intrinsicMatrix = K^T and R = R^T (MATLAB/JARVIS style), exactly
      what athena.labels2d.read_calibration() expects.

World frame (defined by the board lying on the table):
    * origin: inner corner of the board adjacent to the RED edge (or board centre, --origin center)
    * X: along the long side of the board (parallel to the red edge)
    * Y: along the short side, pointing away from the red edge
    * Z: up, towards the cameras

The board in these recordings is 5x7 squares (4x6 inner corners), which is symmetric
under a 180 deg rotation: OpenCV may return the corners in opposite order in different
cameras. The red edge painted on one long side of the board is used to make the corner
ordering (and therefore the world frame) consistent across all cameras.

Intrinsics: with a single board view per camera, the focal length is estimated starting
from the nominal lens focal length / pixel pitch, with principal point fixed at the image
centre. Distortion can only be roughly estimated from a single view; for best accuracy
calibrate intrinsics once with many board poses and pass them with --intrinsics-dir.

ATHENA pairs calibration files and videos by ALPHABETICAL ORDER: put the videos in
<out>/videos/<recording>/ named like the calibration files (cam_1.avi ... cam_9.avi).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PIXEL_SIZE_UM = 5.86  # Basler acA1920-40gc (Sony IMX249), 1920x1200


# --------------------------------------------------------------------------------------
# Input discovery
# --------------------------------------------------------------------------------------

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def find_frames_dir(rec_dir, explicit=None):
    if explicit:
        d = Path(explicit)
        if not d.is_dir():
            sys.exit(f'Calibration frames folder not found: {d}')
        return d
    candidates = sorted(p for p in (rec_dir / 'calibration_frames').glob('*') if p.is_dir())
    if not candidates:
        sys.exit(f'No calibration_frames/<capture> folder in {rec_dir}')
    if len(candidates) > 1:
        print(f'[info] {len(candidates)} calibration captures found, using the latest: {candidates[-1].name}')
    return candidates[-1]


def load_camera_config(rec_dir, frames_dir):
    """Return {cam_name: {'serial': str, 'focal_length_mm': float|None, 'flip_camera': str}}"""
    for cfg_path in (frames_dir / 'camera_config.json', rec_dir / 'metadata' / 'camera_config.json'):
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text())
            break
    else:
        src = rec_dir / 'source_calibration.json'
        if not src.is_file():
            sys.exit('No camera_config.json / source_calibration.json found')
        cfg = json.loads(src.read_text())['raw']['config']

    cams = {}
    for c in cfg['cameras']:
        cal = c.get('calibration', {})
        cams[c['name']] = {
            'serial': str(c['serial']),
            'focal_length_mm': cal.get('focal_length_mm'),
            'flip_camera': cal.get('flip_camera'),
        }
    return cams


def list_calibration_images(frames_dir):
    """Return {cam_name: {frame_index: path}}"""
    rx = re.compile(r'^(cam_\d+)_frame(\d+)\.(png|jpe?g|bmp|tiff?)$', re.IGNORECASE)
    images = {}
    for p in frames_dir.iterdir():
        m = rx.match(p.name)
        if m:
            images.setdefault(m.group(1), {})[int(m.group(2))] = p
    if not images:
        sys.exit(f'No cam_<N>_frame<K>.png images in {frames_dir}')
    return images


def find_video(rec_dir, serial):
    vids = sorted(rec_dir.glob(f'*-M{serial}.*'))
    vids = [v for v in vids if v.suffix.lower() in ('.mkv', '.avi', '.mp4', '.mov')]
    return vids[0] if vids else None


# --------------------------------------------------------------------------------------
# Board detection and canonical ordering
# --------------------------------------------------------------------------------------

def detect_board(gray, pattern):
    """Detect inner corners. pattern = (nx, ny) = (corners per row, rows).
    Returns (ny, nx, 2) float64 array or None."""
    nx, ny = pattern
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not ok:
            return None
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return corners.reshape(ny, nx, 2).astype(np.float64)


def signed_area(grid):
    """Shoelace area of the outer-corner quad, image coords (y down)."""
    q = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
    x, y = q[:, 0], q[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def grid_homography(grid):
    """Homography from board grid coordinates (i = column, j = row) to image pixels."""
    ny, nx = grid.shape[:2]
    jj, ii = np.mgrid[0:ny, 0:nx]
    src = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.float64)
    H, _ = cv2.findHomography(src, grid.reshape(-1, 2))
    return H


def redness_of_side(img_bgr, H, side, nx, ny):
    """Max over parallel lines (outside the squares, on one long side of the board)
    of the median (R - G) value. side = 'low' (i < 0) or 'high' (i > nx-1)."""
    h, w = img_bgr.shape[:2]
    red = img_bgr[:, :, 2].astype(np.float32) - img_bgr[:, :, 1].astype(np.float32)
    red = cv2.dilate(red, np.ones((5, 5), np.uint8))  # widen the thin red strip
    js =np.linspace(0.0, ny - 1, 60)
    best = -np.inf
    # outer square edge is at i = -1 (low) / i = nx (high); margin + red strip lie beyond it
    for d in np.linspace(0.05, 1.6, 40):
        i = -1.0 - d if side == 'low' else nx + d
        pts = np.stack([np.full_like(js, i), js], axis=1).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        inside = (px[:, 0] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] >= 0) & (px[:, 1] <= h - 1)
        if inside.sum() < 0.5 * len(js):
            continue
        px = px[inside].astype(np.float32)
        vals = cv2.remap(red, px[:, 0].reshape(-1, 1), px[:, 1].reshape(-1, 1), cv2.INTER_LINEAR)
        best = max(best, float(np.median(vals)))
    return best if np.isfinite(best) else float('nan')


def canonicalize(grid, img_bgr, force_flip=False, min_contrast=15.0):
    """Make the corner ordering identical in every camera:
      1. handedness: (i, j) axes appear right-handed in the image (camera above the board)
      2. 180 deg ambiguity: red edge on the i < 0 side.
    Returns (grid, info dict)."""
    ny, nx = grid.shape[:2]
    info = {'mirrored': False, 'rotated_180': False}
    if signed_area(grid) < 0:
        grid = grid[:, ::-1].copy()
        info['mirrored'] = True

    H = grid_homography(grid)
    r_low = redness_of_side(img_bgr, H, 'low', nx, ny)
    r_high = redness_of_side(img_bgr, H, 'high', nx, ny)
    info['red_score_low_side'] = r_low
    info['red_score_high_side'] = r_high
    rotate = bool(r_high > r_low) if np.isfinite(r_low) and np.isfinite(r_high) else False
    info['red_edge_confident'] = bool(np.isfinite(r_low) and np.isfinite(r_high)
                                      and abs(r_high - r_low) >= min_contrast)
    if force_flip:
        rotate = not rotate
        info['forced_flip'] = True
    if rotate:
        grid = grid[::-1, ::-1].copy()
        info['rotated_180'] = True
    return grid, info


def board_object_points(nx, ny, square, origin='corner'):
    """World coordinates of the inner corners, ordered like the canonical grid (ny, nx).
    Grid column i runs along the short side (away from the red edge) -> world Y,
    grid row j runs along the long side -> world X, Z = X x Y points up."""
    jj, ii = np.mgrid[0:ny, 0:nx]
    obj = np.zeros((ny, nx, 3), np.float64)
    obj[..., 0] = jj * square
    obj[..., 1] = ii * square
    if origin == 'center':
        obj[..., 0] -= (ny - 1) * square / 2
        obj[..., 1] -= (nx - 1) * square / 2
    return obj


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------

def nominal_K(focal_mm, pixel_um, w, h):
    f = focal_mm / (pixel_um * 1e-3)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], np.float64)


def read_athena_yaml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode('intrinsicMatrix').mat().T
    dist = fs.getNode('distortionCoefficients').mat().reshape(-1)
    fs.release()
    return K, dist


def estimate_intrinsics(obj_list, img_list, size, K0, mode):
    """Single/few-view intrinsics refinement around a nominal guess."""
    dist0 = np.zeros(5)
    if mode == 'nominal':
        return K0.copy(), dist0, None
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    if mode == 'focal':
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    elif mode == 'focal_k1':
        flags |= cv2.CALIB_FIX_K2
    elif mode == 'full':
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    obj = [o.reshape(-1, 1, 3).astype(np.float32) for o in obj_list]
    img = [p.reshape(-1, 1, 2).astype(np.float32) for p in img_list]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, K0.copy(), dist0.copy(), flags=flags)
    return K, dist.reshape(-1)[:5], rms


def solve_pose(obj, img, K, dist):
    obj = obj.reshape(-1, 3)
    img = img.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    return rvec.reshape(3), tvec.reshape(3), float(np.sqrt(np.mean(err ** 2))), float(err.max())


def triangulate_dlt(points_2d, cams):
    """points_2d: list over cams of (N,2) pixel arrays; cams: list of dict(K, dist, R, t)."""
    N = points_2d[0].shape[0]
    norm = [cv2.undistortPoints(p.reshape(-1, 1, 2), c['K'], c['dist']).reshape(-1, 2)
            for p, c in zip(points_2d, cams)]
    X = np.zeros((N, 3))
    for k in range(N):
        A = []
        for c, uv in zip(cams, norm):
            P = np.hstack([c['R'], c['t'].reshape(3, 1)])
            A.append(uv[k, 0] * P[2] - P[0])
            A.append(uv[k, 1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X[k] = Vt[-1, :3] / Vt[-1, 3]
    return X


# --------------------------------------------------------------------------------------
# Checks / outputs
# --------------------------------------------------------------------------------------

def video_orientation_check(video_path, calib_img_bgr):
    """Compare the first video frame with the calibration screenshot (same and 180 deg)."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    def prep(im):
        g = cv2.cvtColor(cv2.resize(im, (240, 150), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float64)
        return (g - g.mean()) / (g.std() + 1e-9)

    a = prep(calib_img_bgr)
    b = prep(frame)
    return {
        'video_size': [int(frame.shape[1]), int(frame.shape[0])],
        'ncc_same': float(np.mean(a * b)),
        'ncc_rot180': float(np.mean(a * b[::-1, ::-1])),
    }


def draw_check(img, grid, K, dist, rvec, tvec, square, name, rms, red_info):
    out = img.copy()
    ny, nx = grid.shape[:2]
    cv2.drawChessboardCorners(out, (nx, ny), grid.reshape(-1, 1, 2).astype(np.float32), True)
    cv2.circle(out, tuple(int(v) for v in grid[0, 0]), 14, (255, 0, 255), 3)
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, 2 * square, 4)
    txt = [f'{name}  reproj RMS {rms:.2f}px',
           f'red score low/high side: {red_info["red_score_low_side"]:.1f} / '
           f'{red_info["red_score_high_side"]:.1f}'
           + ('' if red_info['red_edge_confident'] else '  (AMBIGUOUS!)'),
           'axes: X red, Y green, Z blue (up); magenta = origin corner']
    for k, t in enumerate(txt):
        cv2.putText(out, t, (20, 40 + 35 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(out, t, (20, 40 + 35 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def write_athena_yaml(path, K, dist, R, t):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write('intrinsicMatrix', np.asarray(K, np.float64).T)
    fs.write('distortionCoefficients', np.asarray(dist, np.float64).reshape(1, -1))
    fs.write('R', np.asarray(R, np.float64).T)
    fs.write('T', np.asarray(t, np.float64).reshape(3, 1))
    fs.release()


def write_athena_toml(path, cams, metadata):
    """Anipose-style TOML: [cam_0] ... [cam_{n-1}] + exactly one [metadata] table
    (ATHENA counts cameras as len(file) - 1)."""
    def arr(a):
        a = np.asarray(a, np.float64)
        if a.ndim == 1:
            return '[ ' + ', '.join(repr(float(v)) for v in a) + ' ]'
        return '[ ' + ', '.join(arr(r) for r in a) + ' ]'

    lines = []
    for k, c in enumerate(cams):
        lines += [f'[cam_{k}]',
                  f'name = "{c["name"]}"',
                  f'size = [ {c["size"][0]}, {c["size"][1]} ]',
                  f'matrix = {arr(c["K"])}',
                  f'distortions = {arr(c["dist"])}',
                  f'rotation = {arr(c["rvec"])}',
                  f'translation = {arr(c["t"])}',
                  '']
    lines.append('[metadata]')
    for k, v in metadata.items():
        lines.append(f'{k} = {json.dumps(v)}')
    Path(path).write_text('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording_dir', type=Path, help='recording folder (contains calibration_frames/, *.mkv, metadata)')
    ap.add_argument('--square-mm', type=float, default=100.0,
                    help='checkerboard square side in millimetres (default: 100 = lab board, 10 cm squares)')
    ap.add_argument('--inner-corners', default='4x6',
                    help='inner corners "NxM" (short side x long side). Default 4x6 = board of 5x7 squares')
    ap.add_argument('--frames-dir', type=Path, help='calibration screenshots folder (default: latest in calibration_frames/)')
    ap.add_argument('--out', type=Path, help='output ATHENA main folder (default: <recording_dir>/athena)')
    ap.add_argument('--format', choices=['yaml', 'toml'], default='yaml', help='ATHENA calibration format')
    ap.add_argument('--intrinsics-mode', choices=['focal', 'focal_k1', 'nominal', 'full'], default='focal',
                    help='focal: fit focal length only (default); focal_k1: also radial k1; '
                         'nominal: lens focal / pixel pitch, no distortion; full: needs many board poses')
    ap.add_argument('--intrinsics-dir', type=Path,
                    help='folder with pre-computed intrinsics (<cam>.yaml in ATHENA/JARVIS format); '
                         'only extrinsics are computed')
    ap.add_argument('--pixel-size-um', type=float, default=DEFAULT_PIXEL_SIZE_UM, help='sensor pixel pitch (um)')
    ap.add_argument('--max-focal-deviation', type=float, default=0.15,
                    help='if fitted focal differs from nominal more than this fraction, fall back to nominal')
    ap.add_argument('--ref-frame', type=int, default=None,
                    help='frame index used for extrinsics (default: lowest index seen by all cameras)')
    ap.add_argument('--origin', choices=['corner', 'center'], default='corner', help='world origin on the board')
    ap.add_argument('--flip-cams', default='', help='comma separated cameras whose red-edge decision must be inverted')
    ap.add_argument('--skip-video-check', action='store_true', help='do not compare video frames with screenshots')
    return ap.parse_args()


def main():
    args = parse_args()
    rec_dir = args.recording_dir.resolve()
    out_dir = (args.out or rec_dir / 'athena').resolve()
    cal_dir = out_dir / 'calibration'
    chk_dir = out_dir / 'calibration_check'
    cal_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)

    nx, ny = (int(v) for v in args.inner_corners.lower().split('x'))
    pattern = (nx, ny)
    square = args.square_mm
    flip_cams = {c.strip() for c in args.flip_cams.split(',') if c.strip()}

    frames_dir = find_frames_dir(rec_dir, args.frames_dir)
    cam_cfg = load_camera_config(rec_dir, frames_dir)
    images = list_calibration_images(frames_dir)
    cam_names = sorted(images, key=natural_key)
    missing = [c for c in cam_names if c not in cam_cfg]
    if missing:
        sys.exit(f'Cameras without entry in camera_config.json: {missing}')

    print(f'Recording : {rec_dir}')
    print(f'Frames    : {frames_dir}')
    print(f'Board     : {nx}x{ny} inner corners ({nx + 1}x{ny + 1} squares), square {square} mm')
    print(f'Cameras   : {", ".join(cam_names)}\n')

    obj_grid = board_object_points(nx, ny, square, args.origin)

    # ---- detection ----------------------------------------------------------------
    detections = {}   # cam -> {frame: (grid, info)}
    imgs = {}
    for cam in cam_names:
        detections[cam] = {}
        for fidx, path in sorted(images[cam].items()):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f'[warn] cannot read {path}')
                continue
            imgs[(cam, fidx)] = img
            grid = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
            if grid is None:
                print(f'[warn] {cam} frame{fidx}: board NOT detected')
                continue
            grid, info = canonicalize(grid, img, force_flip=cam in flip_cams)
            detections[cam][fidx] = (grid, info)
            flag = '' if info['red_edge_confident'] else '   <-- red edge AMBIGUOUS, check overlay'
            print(f'  {cam} frame{fidx}: detected, red score {info["red_score_low_side"]:6.1f} vs '
                  f'{info["red_score_high_side"]:6.1f}, rotated={info["rotated_180"]}{flag}')

    cams_ok = [c for c in cam_names if detections[c]]
    if len(cams_ok) < 2:
        sys.exit('Board detected in fewer than 2 cameras: cannot calibrate. '
                 'Check --inner-corners (inner corners, not squares!).')
    lost = [c for c in cam_names if c not in cams_ok]
    if lost:
        print(f'\n[ERROR] board not detected in {lost}: these cameras will be MISSING from the calibration,'
              ' and ATHENA requires one calibration per video.')

    if args.ref_frame is None:
        common = set.intersection(*(set(detections[c]) for c in cams_ok))
        if not common:
            sys.exit('No frame index in which all cameras see the board; use --ref-frame')
        ref = min(common)
    else:
        ref = args.ref_frame
    print(f'\nExtrinsics from frame{ref}\n')

    # ---- calibration per camera --------------------------------------------------
    results = []
    for cam in cams_ok:
        if ref not in detections[cam]:
            print(f'[ERROR] {cam}: board not detected in reference frame{ref}, camera skipped')
            continue
        cfg = cam_cfg[cam]
        img_ref = imgs[(cam, ref)]
        h, w = img_ref.shape[:2]
        focal_mm = cfg['focal_length_mm']

        intr_note = ''
        f_fit = None
        if args.intrinsics_dir:
            K, dist = read_athena_yaml(args.intrinsics_dir / f'{cam}.yaml')
            intr_note = f'loaded from {args.intrinsics_dir}'
        else:
            if focal_mm is None:
                sys.exit(f'{cam}: no focal_length_mm in config; use --intrinsics-dir')
            K0 = nominal_K(focal_mm, args.pixel_size_um, w, h)
            frames = sorted(detections[cam])
            K, dist, _ = estimate_intrinsics([obj_grid] * len(frames),
                                             [detections[cam][f][0] for f in frames],
                                             (w, h), K0, args.intrinsics_mode)
            f_fit = float(K[0, 0])
            dev = f_fit / K0[0, 0] - 1
            intr_note = f'{args.intrinsics_mode}: f={f_fit:.0f}px (nominal {K0[0, 0]:.0f}px, {dev:+.1%})'
            if abs(dev) > args.max_focal_deviation:
                print(f'[warn] {cam}: fitted focal {f_fit:.0f}px deviates {dev:+.1%} from nominal; '
                      f'falling back to nominal intrinsics')
                K, dist = K0, np.zeros(5)
                intr_note += ' -> REJECTED, nominal used'

        grid = detections[cam][ref][0]
        rvec, tvec, rms, emax = solve_pose(obj_grid, grid, K, dist)
        R, _ = cv2.Rodrigues(rvec)
        center = (-R.T @ tvec)

        vid = find_video(rec_dir, cfg['serial'])
        vcheck = None
        if vid is not None and not args.skip_video_check:
            vcheck = video_orientation_check(vid, img_ref)
            if vcheck is not None:
                if vcheck['ncc_rot180'] > vcheck['ncc_same'] + 0.1:
                    print(f'[ERROR] {cam}: video frames look ROTATED 180 deg w.r.t. the calibration '
                          f'screenshot (ncc {vcheck["ncc_same"]:.2f} vs {vcheck["ncc_rot180"]:.2f})!')
                elif vcheck['ncc_same'] < 0.5:
                    print(f'[warn] {cam}: first video frame poorly matches the screenshot '
                          f'(ncc {vcheck["ncc_same"]:.2f}); did the camera move?')
                if vcheck['video_size'] != [w, h]:
                    print(f'[ERROR] {cam}: video size {vcheck["video_size"]} != screenshot size {[w, h]}')

        cv2.imwrite(str(chk_dir / f'{cam}.jpg'),
                    draw_check(img_ref, grid, K, dist, rvec, tvec, square, cam, rms, detections[cam][ref][1]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append(dict(name=cam, serial=cfg['serial'], size=[w, h], K=K, dist=dist, rvec=rvec,
                            R=R, t=tvec, center=center, rms=rms, emax=emax, focal_mm=focal_mm,
                            f_fit=f_fit, intr_note=intr_note, video=vid.name if vid else None,
                            vcheck=vcheck, red=detections[cam][ref][1]))
        print(f'  {cam} (serial {cfg["serial"]}): reproj RMS {rms:.2f}px (max {emax:.2f}), '
              f'camera at [{center[0]:7.0f} {center[1]:7.0f} {center[2]:7.0f}] mm, '
              f'{np.linalg.norm(tvec):.0f} mm from origin | {intr_note}')

    # ---- multi-view consistency check -------------------------------------------
    tri_rms = None
    if len(results) >= 2:
        pts = [detections[r['name']][ref][0].reshape(-1, 2) for r in results]
        X = triangulate_dlt(pts, results)
        d = np.linalg.norm(X - obj_grid.reshape(-1, 3), axis=1)
        tri_rms = float(np.sqrt(np.mean(d ** 2)))
        print(f'\nTriangulated board corners vs true board: RMS {tri_rms:.2f} mm, max {d.max():.2f} mm')
        if np.mean([r['center'][2] for r in results]) < 0:
            print('[warn] cameras are BELOW the board (Z<0): unexpected for a board on the table')
        low_conf = [r['name'] for r in results if not r['red']['red_edge_confident']]
        if low_conf:
            print(f'[warn] red edge ambiguous for {low_conf}: open calibration_check/*.jpg and verify that the '
                  f'magenta origin corner is next to the red edge in all views; fix with --flip-cams')

    # ---- write ATHENA calibration -------------------------------------------------
    for old in list(cal_dir.glob('*.yaml')) + list(cal_dir.glob('*.toml')):
        old.unlink()
    metadata = {
        'source': 'athena_calibration.py',
        'recording': rec_dir.name,
        'calibration_frames': frames_dir.name,
        'units': 'mm',
        'square_mm': square,
        'inner_corners': [nx, ny],
        'camera_order': [r['name'] for r in results],
        'serials': [r['serial'] for r in results],
    }
    # ATHENA sorts both calibration files and videos alphabetically -> use that order everywhere
    results_alpha = sorted(results, key=lambda r: r['name'])
    metadata['camera_order'] = [r['name'] for r in results_alpha]
    metadata['serials'] = [r['serial'] for r in results_alpha]
    if args.format == 'yaml':
        for r in results:
            write_athena_yaml(cal_dir / f'{r["name"]}.yaml', r['K'], r['dist'], r['R'], r['t'])
    else:
        write_athena_toml(cal_dir / 'calibration.toml', results_alpha, metadata)

    order = [r['name'] for r in results_alpha]
    if order != [r['name'] for r in results]:
        print('[warn] alphabetical order of calibration files differs from natural order '
              '(e.g. cam_10 < cam_2): name your videos accordingly!')

    report = {
        **metadata,
        'format': args.format,
        'intrinsics_mode': 'loaded' if args.intrinsics_dir else args.intrinsics_mode,
        'reference_frame': ref,
        'world_frame': {'origin': args.origin,
                        'X': 'along long side of the board (parallel to red edge)',
                        'Y': 'along short side, away from red edge',
                        'Z': 'up (towards cameras)'},
        'triangulation_rms_mm': tri_rms,
        'athena_order': [{'index': k, 'camera': n,
                          'serial': next(r['serial'] for r in results if r['name'] == n),
                          'source_video': next(r['video'] for r in results if r['name'] == n),
                          'athena_video_name': f'{n}.avi'} for k, n in enumerate(order)],
        'cameras': [{
            'name': r['name'], 'serial': r['serial'], 'image_size': r['size'],
            'video': r['video'], 'focal_length_mm_nominal': r['focal_mm'],
            'intrinsics': r['intr_note'],
            'K': r['K'].tolist(), 'dist': np.asarray(r['dist']).tolist(),
            'rvec': r['rvec'].tolist(), 'R': r['R'].tolist(), 'T_mm': r['t'].tolist(),
            'camera_center_mm': r['center'].tolist(),
            'reprojection_rms_px': r['rms'], 'reprojection_max_px': r['emax'],
            'board_orientation': r['red'], 'video_check': r['vcheck'],
        } for r in results],
    }
    (out_dir / 'calibration_report.json').write_text(json.dumps(report, indent=2, default=float))

    print(f'\nWritten {len(results)} camera(s) to {cal_dir}')
    print(f'Report : {out_dir / "calibration_report.json"}')
    print(f'Overlays: {chk_dir}  <-- check them!')
    print('\nATHENA pairs calibrations and videos by alphabetical order. Expected videos:')
    for item in report['athena_order']:
        print(f'  [{item["index"]}] {out_dir.name}/videos/<recording>/{item["athena_video_name"]}  '
              f'<- {item["source_video"]}')


if __name__ == '__main__':
    main()
