#!/usr/bin/env python3
"""
Collect all ATHENA 3D skeleton videos (data3d.mp4) into one folder, named after their recording.

    python scripts/collect_data3d.py [ROOT] [--out FOLDER] [--mode copy|symlink] [--select TEXT] [--overwrite]

ROOT (default: data/ next to scripts/) is searched recursively for
    <recording>/athena/videos_processed/<trial>/data3d.mp4
and each video is copied to
    FOLDER/<recording>.mp4                  (default FOLDER: videos3ddata/ next to scripts/)
    FOLDER/<recording>__<trial>.mp4         (only if a recording has more than one trial)

Videos already collected and unchanged are skipped (re-run it after processing new recordings).
"""
import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def recording_name(video):
    """<recording>/athena/videos_processed/<trial>/data3d.mp4 -> (recording, trial)."""
    trial_dir = video.parent
    main = trial_dir.parent.parent  # .../athena (ATHENA main folder)
    rec = main.parent if main.name == 'athena' else main
    return rec.name, trial_dir.name


def matches(name, pattern):
    if any(c in pattern for c in '*?['):
        return fnmatch.fnmatch(name, pattern)
    return pattern.lower() in name.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?', type=Path, default=PROJECT / 'data', help='folder to search (default: data/)')
    ap.add_argument('--out', type=Path, default=PROJECT / 'videos3ddata', help='destination (default: videos3ddata/)')
    ap.add_argument('--mode', choices=('copy', 'symlink'), default='copy',
                    help='copy (default: independent files, ~10 MB each) or symlink (0 bytes, needs the data folder)')
    ap.add_argument('--select', action='append', metavar='TEXT',
                    help='only recordings whose name contains TEXT or matches the wildcard (repeatable)')
    ap.add_argument('--overwrite', action='store_true', help='replace videos already collected even if unchanged')
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f'ERROR: {root} is not a folder')
    out = args.out.resolve()

    videos = sorted(v for v in root.rglob('data3d.mp4')
                    if v.parent.parent.name == 'videos_processed' and out not in v.parents)
    found = {}
    for v in videos:
        rec, trial = recording_name(v)
        if args.select and not any(matches(rec, p) for p in args.select):
            continue
        found.setdefault(rec, []).append((trial, v))
    if not found:
        sys.exit(f'no data3d.mp4 found in {root}' + (f' matching {args.select}' if args.select else ''))

    out.mkdir(parents=True, exist_ok=True)
    n_new = n_skip = 0
    for rec, items in sorted(found.items()):
        for trial, src in items:
            dst = out / (f'{rec}.mp4' if len(items) == 1 else f'{rec}__{trial}.mp4')
            if dst.exists() and not args.overwrite:
                same = (dst.resolve() == src.resolve()) if dst.is_symlink() else (
                    dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime)
                if same:
                    n_skip += 1
                    continue
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            if args.mode == 'symlink':
                dst.symlink_to(os.path.relpath(src, dst.parent))
            else:
                shutil.copy2(src, dst)
            n_new += 1
            print(f'{args.mode:7s} {dst.name}  <-  {src.relative_to(root)}')
    print(f'{n_new} collected, {n_skip} already up to date -> {out}')


if __name__ == '__main__':
    main()
