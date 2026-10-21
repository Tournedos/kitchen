"""Organize multi-camera captures from Dexter/data/ into ATHENA project folders.

Every capture folder in data/ (any folder with the recorder's metadata.json)
becomes one ATHENA "main folder":

    athena_projects/<capture>/
    ├── videos/<capture>/cam0.avi ... cam8.avi   lossless remux of the Take_*.mkv files
    ├── calibration/calibration.toml              camera calibration, same camera order
    ├── calibration_check.jpg                     floor grid drawn into every camera
    ├── calibration_report.json                   per-camera errors, focal lengths, positions
    ├── preview.jpg                               one labelled frame per camera
    └── prep_manifest.json                        camera map (cam0 <-> cam_1 <-> serial) + checks

ATHENA pairs videos and calibration files purely by sorted filename, so both are
written as cam0, cam1, ... in the same order (lab camera names cam_1, cam_2, ...).

Calibration is computed from the checkerboard snapshot the recorder saves with
every capture (see calibrate_rig.py). Set SQUARE_MM below to the measured side
of one checkerboard square to get exact millimetres. To use a calibration made
elsewhere instead, put JARVIS .yaml files (one per camera) or one Anipose .toml
in data/<capture>/calibration/, named/labelled by camera name (cam_1) or serial
(25369878), or pass --calibration DIR for every capture.

Re-running is safe: finished videos and calibrations are skipped.

    conda activate athena
    python prepare_athena.py
"""

import argparse
import json
import math
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
import toml

import calibrate_rig

DEXTER_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = DEXTER_ROOT / "data"
OUT_ROOT = DEXTER_ROOT / "athena_projects"
# Side of one checkerboard square in mm: the squares drawn on the two boards are
# 10 cm (also what the calibration itself finds: counter at ~87 cm, cameras at ~2.4 m).
SQUARE_MM = 100.0


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def find_captures(data_root):
    return sorted(p.parent for p in Path(data_root).glob("*/metadata.json"))


def load_capture(capture_dir):
    """Read the recorder metadata into an ordered camera list plus warnings."""
    meta = json.loads((capture_dir / "metadata.json").read_text())
    static = meta["camera_static_metadata_by_serial"]
    warnings = []

    cameras = []
    for out in meta["camera_outputs"]:
        serial = str(out["serial"])
        cameras.append({
            "camera_name": static[serial]["camera_name"],
            "serial": serial,
            "source_video": capture_dir / out["video_filename"],
            "fps": float(out["frame_rate"]),
            "frames": int(out["frames_written"]),
        })
        if out.get("missing_frames"):
            warnings.append(f"{serial}: recorder reports {out['missing_frames']} missing frames")
    cameras.sort(key=lambda c: natural_key(c["camera_name"]))

    # ATHENA sorts filenames as strings, so cam10 would land before cam2.
    if len(cameras) > 10:
        raise ValueError("more than 10 cameras: cam0..camN names would not sort correctly in ATHENA")
    for i, cam in enumerate(cameras):
        cam["athena_name"] = f"cam{i}"

    if len({c["frames"] for c in cameras}) > 1:
        warnings.append("cameras have different frame counts: " + ", ".join(
            f"{c['camera_name']}={c['frames']}" for c in cameras))
    if len({c["fps"] for c in cameras}) > 1:
        warnings.append("cameras have different frame rates")

    result_file = capture_dir / "capture_result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if not result.get("verification", {}).get("ok", True):
            warnings.append("recorder verification failed (see capture_result.json)")
        sync = result.get("synchronization", {})
        if sync.get("max_timestamp_skew_ms", 0) > sync.get("timestamp_skew_limit_ms", math.inf):
            warnings.append(f"timestamp skew {sync['max_timestamp_skew_ms']} ms exceeds limit")

    return meta, cameras, warnings


# ---------------------------------------------------------------- videos

def avi_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return stream.frames, float(stream.average_rate or 0)


def remux_video(src, dst, fps, expected_frames):
    """Copy the MJPEG stream into an AVI with constant-rate timestamps.

    The MKV carries millisecond timestamps; forcing the input rate gives a
    1/fps time base so the AVI header holds the exact frame count ATHENA reads.
    The recorder already gap-fills at nominal fps, so frames map 1:1.
    """
    tmp = dst.with_suffix(".avi.part")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-r", f"{fps:g}", "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "avi", str(tmp)]
    subprocess.run(cmd, check=True)
    frames, rate = avi_info(tmp)
    if frames != expected_frames or rate != fps:
        tmp.unlink()
        raise RuntimeError(f"{src.name}: remux gave {frames} frames @ {rate} fps, "
                           f"expected {expected_frames} @ {fps}")
    tmp.replace(dst)


def prepare_videos(cameras, video_dir, force=False):
    video_dir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        dst = video_dir / f"{cam['athena_name']}.avi"
        cam["video"] = dst
        if not force and dst.exists() and avi_info(dst) == (cam["frames"], cam["fps"]):
            print(f"    {dst.name}  ok (already done)")
            continue
        if not cam["source_video"].exists():
            raise FileNotFoundError(f"missing video {cam['source_video']}")
        print(f"    {dst.name}  <- {cam['camera_name']} ({cam['source_video'].name})")
        remux_video(cam["source_video"], dst, cam["fps"], cam["frames"])


# ---------------------------------------------------------------- calibration

CALIBRATION_SUFFIXES = {".yaml", ".yml", ".toml"}


def find_calibration_source(capture_dir, fallback=None):
    for candidate in (capture_dir / "calibration", fallback):
        if candidate and any(p.suffix in CALIBRATION_SUFFIXES for p in Path(candidate).glob("*")):
            return Path(candidate)
    return None


def match_cameras(cameras, labelled):
    """Map each camera to the calibration entry labelled with its name or serial.

    labelled: [(label, entry), ...]. A label matches a camera if it equals the
    camera name (cam_1) or contains its serial. Raises if any camera is
    unmatched or ambiguous so a wrong pairing can never be written silently.
    """
    matched = []
    for cam in cameras:
        hits = [entry for label, entry in labelled
                if label == cam["camera_name"] or cam["serial"] in label]
        if len(hits) != 1:
            raise ValueError(
                f"{cam['camera_name']} (serial {cam['serial']}) matched {len(hits)} calibration "
                f"entries; available labels: {sorted(label for label, _ in labelled)}")
        matched.append(hits[0])
    return matched


def check_yaml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    shapes = {key: getattr(fs.getNode(key).mat(), "shape", None)
              for key in ("intrinsicMatrix", "distortionCoefficients", "R", "T")}
    fs.release()
    if shapes["intrinsicMatrix"] != (3, 3) or shapes["R"] != (3, 3) or None in shapes.values():
        raise ValueError(f"{path.name} is not a JARVIS calibration file: {shapes}")


def sync_calibration(src_dir, cameras, dst_dir):
    yaml_files = sorted(src_dir.glob("*.yaml")) + sorted(src_dir.glob("*.yml"))
    toml_files = sorted(src_dir.glob("*.toml"))

    if yaml_files:
        sources = match_cameras(cameras, [(p.stem, p) for p in yaml_files])
        for path in sources:
            check_yaml(path)
        _clear_calibration(dst_dir)
        for cam, path in zip(cameras, sources):
            shutil.copyfile(path, dst_dir / f"{cam['athena_name']}.yaml")
        return "JARVIS yaml"

    if len(toml_files) != 1:
        raise ValueError(f"expected one .toml in {src_dir}, found {len(toml_files)}")
    cal = toml.load(toml_files[0])
    # Match on each section's `name` only: Anipose section keys are just
    # indices (cam_0, cam_1, ...) and cam_1 there is not the lab's cam_1.
    matched = match_cameras(cameras, [
        (str(section.get("name", "")), section) for section in cal.values()
        if isinstance(section, dict) and "matrix" in section])
    # ATHENA reads sections cam_0..cam_{n-1} and counts cameras as len(file) - 1,
    # so the output holds exactly those sections plus [metadata].
    out = {f"cam_{i}": {**section, "name": cam["athena_name"]}
           for i, (cam, section) in enumerate(zip(cameras, matched))}
    out["metadata"] = {**cal.get("metadata", {}), "source": str(toml_files[0])}
    _clear_calibration(dst_dir)
    (dst_dir / "calibration.toml").write_text(toml.dumps(out))
    return "Anipose toml"


def auto_calibrate(capture_dir, cameras, project, square_mm=None, force=False):
    """Calibrate from the capture's checkerboard snapshot (skipped if already done
    with the same square size). Returns the calibration report."""
    dst = project / "calibration"
    report_path = project / "calibration_report.json"
    if not force and report_path.exists() and (dst / "calibration.toml").exists():
        old = json.loads(report_path.read_text())
        if (old["scale_is_provisional"] == (square_mm is None)
                and (square_mm is None or old["square_mm_large_board"] == float(square_mm))):
            return old
    print("    calibrating from the checkerboard snapshot (about a minute)...")
    calib, report, debug = calibrate_rig.calibrate(capture_dir, cameras, square_mm)
    _clear_calibration(dst)
    calibrate_rig.write_toml(calib, cameras, dst / "calibration.toml", report)
    calibrate_rig.draw_check(calib, cameras, debug["images"], project / "calibration_check.jpg")
    report_path.write_text(json.dumps(report, indent=2))
    return report


def check_project(project):
    """ATHENA README step 2: the calibration must load in ATHENA and match the
    videos of every recording one to one, by name and in order.

    Returns a list of problems (empty if all is well).
    """
    from athena.labels2d import read_calibration

    project = Path(project)
    cal_dir = project / "calibration"
    yaml_files, toml_files = sorted(cal_dir.glob("*.yaml")), sorted(cal_dir.glob("*.toml"))
    if yaml_files:  # ATHENA prefers yaml when both exist
        files, ext, names = yaml_files, "*.yaml", [p.stem for p in yaml_files]
    elif toml_files:
        cal = toml.load([str(p) for p in toml_files])
        files, ext = toml_files, "*.toml"
        names = [cal.get(f"cam_{i}", {}).get("name") for i in range(len(cal) - 1)]
    else:
        return [f"no calibration files in {cal_dir}"]

    problems = []
    try:
        extrinsic, intrinsic, dist = read_calibration([str(p) for p in files], ext)
    except Exception as err:
        return [f"ATHENA cannot read the calibration: {err!r}"]
    for i, (E, K) in enumerate(zip(extrinsic, intrinsic)):
        if E.shape != (4, 4) or K.shape != (3, 3) or not (np.isfinite(E).all() and np.isfinite(K).all()):
            problems.append(f"calibration entry {i} has invalid matrices")

    for rec in sorted(p for p in (project / "videos").iterdir() if p.is_dir()):
        videos = sorted(list(rec.glob("*.avi")) + list(rec.glob("*.mp4")))
        stems = [v.stem for v in videos]
        if stems != names:
            problems.append(f"{rec.name}: videos {stems} do not match calibration names {names}")
        if len({avi_info(v) for v in videos}) > 1:
            problems.append(f"{rec.name}: videos differ in frame count or frame rate")
    return problems


def _clear_calibration(dst_dir):
    # Only top-level files: ATHENA's refined/ subfolder is left alone.
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in [*dst_dir.glob("*.yaml"), *dst_dir.glob("*.toml")]:
        path.unlink()


# ---------------------------------------------------------------- preview

def write_preview(cameras, out_path, tile=(640, 400)):
    """Grid of one mid-recording frame per camera, labelled with the mapping."""
    tiles = []
    for cam in cameras:
        with av.open(str(cam["video"])) as container:
            stream = container.streams.video[0]
            container.seek(stream.frames // 2, stream=stream)
            frame = next(container.decode(stream)).to_ndarray(format="bgr24")
        img = cv2.resize(frame, tile)
        label = f"{cam['athena_name']} = {cam['camera_name']} ({cam['serial']})"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(img)
    cols = math.ceil(math.sqrt(len(tiles)))
    tiles += [tiles[0] * 0] * (-len(tiles) % cols)
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- pipeline

def prepare_capture(capture_dir, out_root=OUT_ROOT, calibration=None, force=False, square_mm=SQUARE_MM):
    capture_dir = Path(capture_dir)
    name = capture_dir.name
    project = Path(out_root) / name
    print(f"\n== {name}")
    summary = {"capture": name, "project": str(project), "status": "ok",
               "calibration": "missing", "warnings": []}
    try:
        meta, cameras, warnings = load_capture(capture_dir)
        summary["warnings"] = warnings
        for w in warnings:
            print(f"    WARNING: {w}")

        prepare_videos(cameras, project / "videos" / name, force=force)

        cal_src = find_calibration_source(capture_dir, calibration)
        try:
            if cal_src:
                summary["calibration"] = f"ok: {sync_calibration(cal_src, cameras, project / 'calibration')} from {cal_src}"
            else:
                report = auto_calibrate(capture_dir, cameras, project, square_mm, force)
                worst = max(c["reprojection_rms_px"] for c in report["cameras"])
                summary["calibration"] = f"ok: from checkerboards, worst camera {worst:.2f} px"
                if report["scale_is_provisional"]:
                    summary["calibration"] += ", 100 mm squares ASSUMED (set SQUARE_MM)"
        except (ValueError, RuntimeError) as err:
            summary["calibration"] = f"error: {err}"
        print(f"    calibration: {summary['calibration']}")

        write_preview(cameras, project / "preview.jpg")
        manifest = {
            "capture": name,
            "source_dir": str(capture_dir),
            "prepared_at": datetime.now().isoformat(timespec="seconds"),
            "recording": {k: meta.get(k) for k in ("start_time_utc", "end_time_utc", "saved_sets")},
            "fps": cameras[0]["fps"],
            "cameras": [{
                "athena_name": c["athena_name"],
                "camera_name": c["camera_name"],
                "serial": c["serial"],
                "frames": c["frames"],
                "video": str(c["video"].relative_to(project)),
                "source_video": c["source_video"].name,
            } for c in cameras],
            "calibration": summary["calibration"],
            "warnings": warnings,
        }
        (project / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))
    except Exception as err:  # keep going with the other captures
        summary["status"] = f"ERROR: {err}"
        print(f"    {summary['status']}")
    return summary


def prepare_all(data_root=DATA_ROOT, out_root=OUT_ROOT, calibration=None, only=None, force=False,
                square_mm=SQUARE_MM):
    captures = find_captures(data_root)
    if only:
        captures = [c for c in captures if c.name in only]
    print(f"Found {len(captures)} capture(s) in {data_root}")
    results = [prepare_capture(c, out_root, calibration, force, square_mm) for c in captures]
    print_summary(results)
    return results


def print_summary(results):
    print("\nSummary")
    width = max([len(r["capture"]) for r in results] + [7])
    for r in results:
        cal = r["calibration"].split(" from /")[0]
        print(f"  {r['capture']:<{width}}  {r['status']}  | calibration {cal}"
              + (f"  | {len(r['warnings'])} warning(s)" if r["warnings"] else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="folder containing the capture folders")
    parser.add_argument("--out", type=Path, default=OUT_ROOT, help="where the ATHENA project folders go")
    parser.add_argument("--calibration", type=Path,
                        help="calibration folder used for captures without data/<capture>/calibration/")
    parser.add_argument("--capture", nargs="+", help="only process these capture folder names")
    parser.add_argument("--square-mm", type=float, default=SQUARE_MM,
                        help="side of one checkerboard square in mm (default: SQUARE_MM in this file)")
    parser.add_argument("--force", action="store_true", help="redo videos and calibration even if already done")
    args = parser.parse_args()
    prepare_all(args.data, args.out, args.calibration, args.capture, args.force, args.square_mm)


if __name__ == "__main__":
    main()
