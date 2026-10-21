"""Calibrate the camera rig from the checkerboard snapshot saved with each capture.

The recorder stores one image per camera (calibration_frames/*/cam_N_frame1.png)
showing two checkerboards of 7x5 squares. From those images this estimates each
camera's focal length, radial distortion and pose:

1. detect every board (6x4 inner corners) in every image
2. decide which board is which, and each board's 180-degree orientation (the
   pattern looks the same rotated):
   - cameras seeing both boards: try every assignment with a small bundle
     adjustment against a reference camera; only the right one fits
   - cameras seeing one board: keep the candidate pose that agrees with SIFT
     feature matches to the cameras already solved
3. bundle-adjust everything: cameras, the second board's pose, the size ratio,
   then again adding SIFT matches between cameras (all images are taken at
   the same instant, so the whole scene acts as a calibration object)
4. express the result in mm with the origin on the floor-most board, z up

Only the board square size is needed from outside (to get millimetres).
"""

import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import toml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PATTERN = (6, 4)          # inner corners per row, per column
PIXEL_PITCH_MM = 0.00586  # Basler ace 1920x1200 (Sony IMX174/IMX249)
GRID = np.array([[i, j, 0] for j in range(PATTERN[1]) for i in range(PATTERN[0])], np.float64)


# ---------------------------------------------------------------- detection

def detect_boards(gray, max_boards=3):
    """All 6x4 checkerboards in an image, as (4, 6, 2) corner arrays.

    Corners are reordered so every board has the same handedness, i.e. we
    always look at its printed face; the 180-degree ambiguity remains.
    """
    boards, work = [], gray.copy()
    for _ in range(max_boards):
        ok, corners = cv2.findChessboardCornersSB(
            work, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            break
        c = corners.reshape(PATTERN[1], PATTERN[0], 2).astype(np.float64)
        along_i, along_j = c[0, 1] - c[0, 0], c[1, 0] - c[0, 0]
        if along_i[0] * along_j[1] - along_i[1] * along_j[0] < 0:
            c = c[:, ::-1]
        boards.append(c)
        # Hide this board (plus its white border) and look for another one.
        pts = c.reshape(-1, 2)
        centre = pts.mean(0)
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        cv2.fillConvexPoly(work, ((hull - centre) * 1.7 + centre).astype(np.int32), 255)
    return boards


def flipped(corners, flip):
    return corners[::-1, ::-1] if flip else corners


# ---------------------------------------------------------------- geometry helpers

def to_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.ravel(rvec)).as_matrix()
    T[:3, 3] = np.ravel(tvec)
    return T


def to_vec(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def board_in_camera(K, corners, flip):
    """Pose of a board (in units of its own squares) in camera coordinates."""
    ok, rvec, tvec = cv2.solvePnP(GRID, flipped(corners, flip).reshape(-1, 1, 2), K, None,
                                  flags=cv2.SOLVEPNP_IPPE)
    return to_matrix(rvec, tvec)


def fundamental(K_a, T_a, K_b, T_b):
    """F such that x_b^T F x_a = 0, for world->camera transforms T_a, T_b."""
    T = T_b @ np.linalg.inv(T_a)
    R, t = T[:3, :3], T[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(K_b).T @ tx @ R @ np.linalg.inv(K_a)


def epipolar_distance(F, pts_a, pts_b):
    """Symmetric point-to-epipolar-line distance in pixels."""
    a = np.c_[pts_a, np.ones(len(pts_a))]
    b = np.c_[pts_b, np.ones(len(pts_b))]
    lines_b, lines_a = a @ F.T, b @ F
    num = np.abs(np.sum(b * lines_b, axis=1))
    return 0.5 * num * (1 / np.hypot(*lines_b[:, :2].T) + 1 / np.hypot(*lines_a[:, :2].T))


class FeatureMatcher:
    """SIFT matches between camera images, computed once per pair."""

    def __init__(self, images):
        sift = cv2.SIFT_create(nfeatures=8000)
        self.features = [sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                         for img in images]
        self.cache = {}

    def __call__(self, a, b):
        if (a, b) not in self.cache:
            (kp_a, des_a), (kp_b, des_b) = self.features[a], self.features[b]
            pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
            good = [m for m, n in pairs if m.distance < 0.8 * n.distance]
            self.cache[(a, b)] = (np.array([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2),
                                  np.array([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2))
        return self.cache[(a, b)]


# ---------------------------------------------------------------- rig model

class Rig:
    """Everything being estimated.

    World frame = board B0, unit = one B0 square. Per camera: world->camera
    rotation vector and translation, focal length, radial k1 (principal point
    fixed at the image centre; k2 overfits with so few board views). B1: its
    pose in the world and the log of its square size relative to B0.
    """

    def __init__(self, f_prior, centre):
        self.f_prior, self.centre = f_prior, centre
        self.cams = {}                 # cam -> [rx, ry, rz, tx, ty, tz, f, k1]
        self.b1 = np.zeros(7)          # [rx, ry, rz, tx, ty, tz, log_ratio]

    def copy(self):
        rig = Rig(self.f_prior, self.centre)
        rig.cams = {c: p.copy() for c, p in self.cams.items()}
        rig.b1 = self.b1.copy()
        return rig

    def set_camera(self, cam, T, f=None):
        self.cams[cam] = np.r_[to_vec(T), f or self.f_prior[cam], 0.0]

    def T(self, cam):
        return to_matrix(self.cams[cam][:3], self.cams[cam][3:6])

    def K(self, cam):
        f = self.cams[cam][6]
        return np.array([[f, 0, self.centre[0]], [0, f, self.centre[1]], [0, 0, 1]])

    def dist(self, cam):
        return np.array([self.cams[cam][7], 0.0, 0.0, 0.0, 0.0])

    @property
    def ratio(self):
        return float(np.exp(self.b1[6]))

    def board_pose(self, board):
        return np.eye(4) if board == 0 else to_matrix(self.b1[:3], self.b1[3:6])

    def board_points(self, board):
        if board == 0:
            return GRID
        B1 = self.board_pose(1)
        return (B1[:3, :3] @ (GRID * self.ratio).T).T + B1[:3, 3]

    def reprojection(self, obs):
        """Per-camera residual vectors (pixels) for observations {cam: [(board, corners)]}."""
        pts = {b: self.board_points(b) for b in (0, 1)}
        out = {}
        for cam, items in obs.items():
            p = self.cams[cam]
            out[cam] = np.concatenate([
                (cv2.projectPoints(pts[b], p[:3], p[3:6], self.K(cam), self.dist(cam))[0].reshape(-1, 2)
                 - corners.reshape(-1, 2)).ravel() for b, corners in items])
        return out

    def undistort(self, cam, pts):
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K(cam), self.dist(cam),
                                   P=self.K(cam)).reshape(-1, 2)


def adjust(rig, obs, free, boards=True, distortion=True, matches=()):
    """Bundle adjustment of the `free` cameras (and B1 if `boards`), in place.

    Residuals: board corner reprojection errors, plus the epipolar distance of
    each SIFT match (a, b, pts_a, pts_b) if given; these cover the whole image
    and pin down distortion far better than the boards alone. Priors: focal
    length within ~15% of the lens spec, small k1, and cameras with the same
    lens spec within ~2% of each other (small far boards cannot separate focal
    length from distance well). Returns per-camera board RMS.
    """
    same_lens = [[c for c in rig.cams if rig.f_prior[c] == f] for f in set(rig.f_prior)]
    same_lens = [g for g in same_lens if len(g) > 1 and set(g) & set(free)]
    free = sorted(free)
    n = 8 if distortion else 7
    x0 = np.concatenate([rig.cams[c][:n] for c in free] + ([rig.b1] if boards else []))

    def apply(x):
        for i, c in enumerate(free):
            rig.cams[c][:n] = x[i * n:(i + 1) * n]
        if boards:
            rig.b1 = x[len(free) * n:].copy()

    def residuals(x):
        apply(x)
        res = list(rig.reprojection(obs).values())
        for a, b, pts_a, pts_b in matches:
            F = fundamental(rig.K(a), rig.T(a), rig.K(b), rig.T(b))
            res.append(epipolar_distance(F, rig.undistort(a, pts_a), rig.undistort(b, pts_b)))
        for c in free:
            f, k1 = rig.cams[c][6:8]
            res.append([(f - rig.f_prior[c]) / (0.15 * rig.f_prior[c]), k1 / 0.5])
        for group in same_lens:
            f = np.array([rig.cams[c][6] for c in group])
            res.append((f - f.mean()) / (0.02 * f.mean()))
        return np.concatenate(res)

    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, x_scale="jac")
    apply(sol.x)
    return {c: float(np.sqrt(np.mean(r ** 2))) for c, r in rig.reprojection(obs).items()}


# ---------------------------------------------------------------- solving

def solve(detections, f_prior, centre, matcher, log):
    """Assign detections to boards, then bundle-adjust. Returns (rig, observations)."""
    two = [c for c, d in enumerate(detections) if len(d) == 2]
    if not two:
        raise RuntimeError("no camera sees both boards; cannot relate them")
    if any(len(d) == 0 for d in detections):
        raise RuntimeError("cameras without any board: "
                           + str([c for c, d in enumerate(detections) if not d]))
    K0 = lambda c: np.array([[f_prior[c], 0, centre[0]], [0, f_prior[c], centre[1]], [0, 0, 1]])

    # Reference camera: sees both boards, largest in the image. Its first
    # detection defines B0 (the world) and the orientation of both boards.
    area = lambda c: sum(cv2.contourArea(b.reshape(-1, 2).astype(np.float32)) for b in detections[c])
    ref = max(two, key=area)
    obs = {ref: [(0, detections[ref][0]), (1, detections[ref][1])]}
    P_ref = [board_in_camera(K0(ref), d, False) for d in detections[ref]]

    rig = Rig(f_prior, centre)
    rig.set_camera(ref, P_ref[0])

    # Cameras seeing both boards: 8 possible assignments; fit each jointly
    # with the reference camera and keep the one that reprojects well.
    pair_fits = []
    for cam in two:
        if cam == ref:
            continue
        hyps = []
        for order in ((0, 1), (1, 0)):
            for f0 in (False, True):
                for f1 in (False, True):
                    c0 = flipped(detections[cam][order[0]], f0)
                    c1 = flipped(detections[cam][order[1]], f1)
                    P0, P1 = board_in_camera(K0(cam), c0, False), board_in_camera(K0(cam), c1, False)
                    via0 = P_ref[0] @ np.linalg.inv(P0)
                    via1 = P_ref[1] @ np.linalg.inv(P1)
                    ratio = np.linalg.norm(via0[:3, 3]) / np.linalg.norm(via1[:3, 3])
                    trial = Rig(f_prior, centre)
                    trial.set_camera(ref, P_ref[0])
                    trial.set_camera(cam, P0)
                    P1_scaled = P_ref[1].copy()
                    P1_scaled[:3, 3] *= ratio
                    trial.b1 = np.r_[to_vec(np.linalg.inv(P_ref[0]) @ P1_scaled), np.log(ratio)]
                    pair_obs = {ref: obs[ref], cam: [(0, c0), (1, c1)]}
                    rms = adjust(trial, pair_obs, {ref, cam}, distortion=False)
                    hyps.append((max(rms.values()), pair_obs[cam], trial))
        hyps.sort(key=lambda h: h[0])
        log(f"    cam{cam}: sees both boards, fit {hyps[0][0]:.2f} px (next best {hyps[1][0]:.2f} px)")
        if hyps[0][0] > 3 or hyps[1][0] < 3 * hyps[0][0]:
            raise RuntimeError(f"cam{cam}: cannot tell which board is which")
        obs[cam] = hyps[0][1]
        rig.cams[cam] = hyps[0][2].cams[cam]
        pair_fits.append(hyps[0][2])
    rig.b1 = np.median([t.b1 for t in pair_fits], axis=0) if pair_fits else rig.b1
    adjust(rig, obs, set(obs), distortion=False)

    # Cameras seeing one board: 4 candidate poses (which board x orientation).
    # Fit the camera to the board under each, then count SIFT matches with the
    # solved cameras that satisfy the resulting epipolar geometry.
    for cam, dets in enumerate(detections):
        if cam in obs:
            continue
        hyps = []
        for board in (0, 1):
            for flip in (False, True):
                corners = flipped(dets[0], flip)
                P = board_in_camera(K0(cam), corners, False)
                P[:3, 3] *= rig.ratio if board == 1 else 1.0
                trial = rig.copy()
                trial.set_camera(cam, P @ np.linalg.inv(rig.board_pose(board)))
                adjust(trial, {cam: [(board, corners)]}, {cam}, boards=False, distortion=False)
                support = 0
                for other in obs:
                    pa, pb = matcher(cam, other)
                    if len(pa):
                        F = fundamental(trial.K(cam), trial.T(cam), trial.K(other), trial.T(other))
                        support += int(np.sum(epipolar_distance(F, pa, trial.undistort(other, pb)) < 8))
                hyps.append((support, [(board, corners)], trial.cams[cam]))
        hyps.sort(key=lambda h: -h[0])
        log(f"    cam{cam}: sees one board, {hyps[0][0]} SIFT matches agree (next best {hyps[1][0]})")
        if hyps[0][0] < 30 or hyps[0][0] < 2 * hyps[1][0]:
            raise RuntimeError(f"cam{cam}: cannot tell which board it sees / its orientation")
        obs[cam] = hyps[0][1]
        rig.cams[cam] = hyps[0][2]

    adjust(rig, obs, set(obs), distortion=False)
    # Refine with the scene itself: SIFT matches that agree with the current
    # estimate, re-selected more strictly once the estimate has improved.
    for thresh in (10.0, 4.0):
        matches = consistent_matches(rig, matcher, len(detections), thresh)
        adjust(rig, obs, set(obs), distortion=True, matches=matches)
    return rig, obs


def consistent_matches(rig, matcher, ncams, thresh, min_count=20):
    """SIFT matches within `thresh` px of their epipolar line, per camera pair."""
    out = []
    for a, b in combinations(range(ncams), 2):
        pa, pb = matcher(a, b)
        if len(pa) < min_count:
            continue
        d = epipolar_distance(fundamental(rig.K(a), rig.T(a), rig.K(b), rig.T(b)),
                              rig.undistort(a, pa), rig.undistort(b, pb))
        keep = d < thresh
        if keep.sum() >= min_count:
            out.append((a, b, pa[keep], pb[keep]))
    return out


# ---------------------------------------------------------------- world frame & output

def world_to_floor(rig, unit_mm):
    """world->camera transforms with the origin on the floor-most board, z up, in mm.

    Both boards lie below the ceiling cameras; the one farthest from them
    along its own normal is the floor. unit_mm: size of one B0 square.
    """
    centres = np.array([np.linalg.inv(rig.T(c))[:3, 3] for c in rig.cams])
    height = {}
    for b in (0, 1):
        B = rig.board_pose(b)
        height[b] = float(np.mean((centres - B[:3, 3]) @ B[:3, 2]))
    floor = max(height, key=lambda b: abs(height[b]))
    floor_to_world = rig.board_pose(floor)
    if height[floor] < 0:  # board z points away from the cameras: turn it over
        floor_to_world = floor_to_world @ np.diag([1, -1, -1, 1.0])
    out = {}
    for c in rig.cams:
        T = rig.T(c) @ floor_to_world
        T[:3, 3] *= unit_mm
        out[c] = T
    return out, floor


def find_snapshot(capture_dir):
    snaps = sorted(Path(capture_dir).glob("calibration_frames/*/"))
    return snaps[-1] if snaps else None


def calibrate(capture_dir, cameras, square_mm=None, log=print):
    """Calibrate `cameras` (dicts with camera_name, in ATHENA order) from the snapshot.

    square_mm: side of one square of the LARGER board, in mm. If None, 100 mm
    is assumed and the report marks the scale as provisional.
    Returns (list of {K, dist, T, size} per camera, report dict, debug data).
    """
    snap = find_snapshot(capture_dir)
    if snap is None:
        raise RuntimeError("no calibration_frames/ snapshot in this capture")
    config = json.loads((snap / "camera_config.json").read_text())
    focal_mm = {c["name"]: c["calibration"]["focal_length_mm"] for c in config["cameras"]}

    images = []
    for cam in cameras:
        img = cv2.imread(str(snap / f"{cam['camera_name']}_frame1.png"))
        if img is None:
            raise RuntimeError(f"missing snapshot image for {cam['camera_name']}")
        images.append(img)
    h, w = images[0].shape[:2]
    detections = [detect_boards(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
    log("    boards found: " + ", ".join(f"{c['camera_name']}={len(d)}" for c, d in zip(cameras, detections)))
    if any(len(d) > 2 for d in detections):
        raise RuntimeError("more than two boards in an image; only two are supported")

    f_prior = [focal_mm[c["camera_name"]] / PIXEL_PITCH_MM for c in cameras]
    matcher = FeatureMatcher(images)
    rig, obs = solve(detections, f_prior, ((w - 1) / 2, (h - 1) / 2), matcher, log)
    rms = {c: float(np.sqrt(np.mean(r ** 2))) for c, r in rig.reprojection(obs).items()}
    check = {}
    for a, b, pa, pb in consistent_matches(rig, matcher, len(cameras), thresh=4.0):
        d = epipolar_distance(fundamental(rig.K(a), rig.T(a), rig.K(b), rig.T(b)),
                              rig.undistort(a, pa), rig.undistort(b, pb))
        check[(a, b)] = (len(d), float(np.median(d)))

    # square_mm is for the larger board; the world unit is one B0 square.
    provisional = square_mm is None
    large_mm = 100.0 if provisional else float(square_mm)
    unit_mm = large_mm / rig.ratio if rig.ratio > 1 else large_mm
    transforms, floor = world_to_floor(rig, unit_mm)
    larger = 1 if rig.ratio > 1 else 0

    calib = [{"K": rig.K(i), "dist": rig.dist(i), "T": transforms[i], "size": [w, h]}
             for i in range(len(cameras))]
    report = {
        "snapshot": str(snap),
        "square_mm_large_board": large_mm,
        "square_mm_small_board": round(large_mm / max(rig.ratio, 1 / rig.ratio), 2),
        "scale_is_provisional": provisional,
        "floor_board": "large" if floor == larger else "small",
        "cameras": [{
            "camera_name": cam["camera_name"],
            "boards_seen": len(obs[i]),
            "reprojection_rms_px": round(rms[i], 2),
            "focal_px": round(float(rig.cams[i][6]), 1),
            "focal_px_lens_spec": round(f_prior[i], 1),
            "k1": round(float(rig.cams[i][7]), 4),
            "position_mm": np.round(-transforms[i][:3, :3].T @ transforms[i][:3, 3], 0).tolist(),
        } for i, cam in enumerate(cameras)],
        "sift_check": {f"{cameras[a]['camera_name']}-{cameras[b]['camera_name']}":
                       {"matches_consistent": n, "median_error_px": round(e, 2)}
                       for (a, b), (n, e) in check.items()},
    }
    return calib, report, {"images": images, "obs": obs, "rig": rig}


def draw_check(calib, cameras, images, out_path, step_mm=500, tile=(640, 400)):
    """Each camera image with a floor grid (every step_mm) and the world axes drawn in.

    With a correct calibration the grid lies flat on the floor and the axes
    (x red, y green, z blue, 1 m long) sit at the same physical spot in every view.
    """
    centres = np.array([-c["T"][:3, :3].T @ c["T"][:3, 3] for c in calib])
    lo = np.floor(centres[:, :2].min(0) / step_mm) * step_mm - step_mm
    hi = np.ceil(centres[:, :2].max(0) / step_mm) * step_mm + step_mm
    lines = []
    for x in np.arange(lo[0], hi[0] + 1, step_mm):
        lines.append(((180, 180, 180), np.c_[np.full(200, x), np.linspace(lo[1], hi[1], 200), np.zeros(200)]))
    for y in np.arange(lo[1], hi[1] + 1, step_mm):
        lines.append(((180, 180, 180), np.c_[np.linspace(lo[0], hi[0], 200), np.full(200, y), np.zeros(200)]))
    for axis, colour in enumerate([(0, 0, 255), (0, 200, 0), (255, 0, 0)]):
        pts = np.zeros((50, 3))
        pts[:, axis] = np.linspace(0, 1000, 50)
        lines.append((colour, pts))

    tiles = []
    for c, cam, img in zip(calib, cameras, images):
        canvas = img.copy()
        rvec, tvec = to_vec(c["T"])[:3], c["T"][:3, 3]
        # Only draw within the field of view: outside it the distortion
        # polynomial folds far-away points back into the image.
        max_radius = 1.1 * np.hypot(*c["K"][:2, 2]) / c["K"][0, 0]
        for colour, pts in lines:
            X = c["T"][:3, :3] @ pts.T + tvec[:, None]
            inside = (X[2] > 100) & (np.hypot(X[0], X[1]) < max_radius * X[2])
            proj = cv2.projectPoints(pts, rvec, tvec, c["K"], c["dist"])[0].reshape(-1, 2)
            for p, q, ok in zip(proj[:-1], proj[1:], inside[:-1] & inside[1:]):
                if ok:
                    cv2.line(canvas, tuple(map(int, p)), tuple(map(int, q)), colour,
                             2 if colour == (180, 180, 180) else 5, cv2.LINE_AA)
        cv2.putText(canvas, cam["camera_name"], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        cv2.putText(canvas, cam["camera_name"], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        tiles.append(cv2.resize(canvas, tile))
    cols = int(np.ceil(np.sqrt(len(tiles))))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(out_path), cv2.vconcat([cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_toml(calib, cameras, path, report):
    """Anipose-style calibration.toml, cameras in the given order (cam_0, cam_1, ...).

    Each camera's name is its ATHENA video name (cam0, cam1, ...), as ATHENA's
    README requires names to match the video filenames.
    """
    out = {}
    for i, (c, cam) in enumerate(zip(calib, cameras)):
        vec = to_vec(c["T"])
        out[f"cam_{i}"] = {"name": cam["athena_name"], "size": c["size"],
                           "matrix": c["K"].tolist(), "distortions": c["dist"].tolist(),
                           "rotation": vec[:3].tolist(), "translation": vec[3:].tolist()}
    out["metadata"] = {"source": "calibrate_rig.py", "units": "mm",
                       "scale_is_provisional": report["scale_is_provisional"]}
    Path(path).write_text(toml.dumps(out))
