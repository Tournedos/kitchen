#!/usr/bin/env python3
"""
Hand joint angles from ATHENA 3D landmarks.

Landmark order (ATHENA): 0-32 body (MediaPipe Pose), 33-53 right hand, 54-74 left hand (MediaPipe Hands:
0 wrist, 1-4 thumb CMC/MCP/IP/tip, 5-8 index MCP/PIP/DIP/tip, 9-12 middle, 13-16 ring, 17-20 pinky).

All angles in degrees. Conventions:
  * finger flexion (MCP, PIP, DIP): angle between proximal and distal segment, measured around the hand's
    lateral (flexion) axis = index MCP -> pinky MCP direction. 0 = straight, + = flexion (towards the palm),
    - = hyperextension.
  * total active motion (TAM) = MCP + PIP + DIP flexion of a finger.
  * finger spread: angle between the proximal phalanges of two adjacent fingers, projected on the palm plane.
  * thumb MCP / IP flexion: 0 = straight (unsigned: the thumb does not flex around the palm axis).
  * thumb abduction: angle between the 1st (thumb) and 2nd (index) metacarpal.
  * wrist flexion/extension: forearm (elbow -> wrist) vs hand (wrist -> middle MCP) around the flexion axis;
    + = palmar flexion, - = extension.
  * wrist radial/ulnar deviation: same vectors in the palm plane; + = radial (thumb side), - = ulnar.
  * forearm rotation: angle between the palm normal and the vertical, around the forearm axis (world Z must be up,
    as in athena_calibration.py): 0 = palm down (pronation), 90 = palm vertical, 180 = palm up (supination).
The wrist and forearm angles use the elbow from MediaPipe Pose, usually less accurate than the hand landmarks.

    from hand_angles import hand_angles
    angles, landmarks = hand_angles(d3, 'R')      # {name: (frames,)}, {name: [landmark indices used]}
"""
import numpy as np

HAND_OFFSET = {'R': 33, 'L': 54}
SIDE_SIGN = {'R': 1.0, 'L': -1.0}       # makes the flexion axis point the same anatomical way for both hands
ELBOW = {'R': 14, 'L': 13}              # MediaPipe Pose: 13 left elbow, 14 right elbow
FINGERS = {'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12), 'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}
THUMB = (1, 2, 3, 4)                    # CMC, MCP, IP, tip
UP = np.array([0.0, 0.0, 1.0])          # world vertical (calibration: Z up)


def _unit(v):
    with np.errstate(invalid='ignore', divide='ignore'):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _proj(v, axis):
    """Component of v perpendicular to the unit vector axis."""
    return v - (v * axis).sum(-1, keepdims=True) * axis


def _signed_angle(u, v, axis):
    """Signed angle [deg] from u to v around axis (both projected on the plane perpendicular to axis)."""
    up, vp = _proj(u, axis), _proj(v, axis)
    return np.degrees(np.arctan2((np.cross(up, vp) * axis).sum(-1), (up * vp).sum(-1)))


def _angle(u, v):
    """Unsigned angle [deg] between u and v."""
    with np.errstate(invalid='ignore', divide='ignore'):
        c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def hand_frame(H, side):
    """Per-frame hand axes from the hand landmarks H (frames, 21, 3):
    lat = flexion axis (positive flexion = towards the palm), lon = wrist -> middle MCP, nrm = palmar normal."""
    lat = _unit((H[:, 5] - H[:, 17]) * SIDE_SIGN[side])
    lon = _unit(H[:, 9] - H[:, 0])
    nrm = _unit(np.cross(lat, lon))
    return lat, lon, nrm


def hand_angles(d3, side):
    """All joint angles of one hand. d3: (frames, 75, 3) ATHENA 3D; side: 'R' or 'L'.
    Returns ({angle name: (frames,) degrees}, {angle name: [global landmark indices involved]})."""
    o = HAND_OFFSET[side]
    H = d3[:, o:o + 21].astype(float)
    g = lambda *idx: [o + i for i in idx]
    lat, lon, nrm = hand_frame(H, side)
    frame_lms = g(0, 5, 9, 17)
    A, L = {}, {}

    def add(name, values, lms):
        A[f'{side} {name}'] = values
        L[f'{side} {name}'] = sorted(set(lms))

    # fingers: flexion at MCP, PIP, DIP (+ TAM)
    for f, (mcp, pip, dip, tip) in FINGERS.items():
        meta, prox, mid, dist = H[:, mcp] - H[:, 0], H[:, pip] - H[:, mcp], H[:, dip] - H[:, pip], H[:, tip] - H[:, dip]
        a_mcp = _signed_angle(meta, prox, lat)
        a_pip = _signed_angle(prox, mid, lat)
        a_dip = _signed_angle(mid, dist, lat)
        add(f'{f} MCP flexion', a_mcp, g(0, mcp, pip) + frame_lms)
        add(f'{f} PIP flexion', a_pip, g(mcp, pip, dip) + frame_lms)
        add(f'{f} DIP flexion', a_dip, g(pip, dip, tip) + frame_lms)
        add(f'{f} TAM', a_mcp + a_pip + a_dip, g(0, mcp, pip, dip, tip) + frame_lms)

    # finger spread (adjacent proximal phalanges on the palm plane)
    names = list(FINGERS)
    for f1, f2 in zip(names[:-1], names[1:]):
        (m1, p1, _, _), (m2, p2, _, _) = FINGERS[f1], FINGERS[f2]
        add(f'spread {f1}–{f2}', _angle(_proj(H[:, p1] - H[:, m1], nrm), _proj(H[:, p2] - H[:, m2], nrm)),
            g(m1, p1, m2, p2) + frame_lms)

    # thumb
    cmc, mcp, ip, tip = THUMB
    add('thumb MCP flexion', _angle(H[:, mcp] - H[:, cmc], H[:, ip] - H[:, mcp]), g(cmc, mcp, ip))
    add('thumb IP flexion', _angle(H[:, ip] - H[:, mcp], H[:, tip] - H[:, ip]), g(mcp, ip, tip))
    add('thumb abduction', _angle(H[:, mcp] - H[:, cmc], H[:, 5] - H[:, 0]), g(cmc, mcp, 0, 5))

    # wrist and forearm (need the elbow from MediaPipe Pose)
    fore = H[:, 0] - d3[:, ELBOW[side]].astype(float)
    hand_dir = H[:, 9] - H[:, 0]
    add('wrist flexion(+)/extension(-)', _signed_angle(fore, hand_dir, lat), [ELBOW[side]] + g(0, 9) + frame_lms)
    e1 = _unit(_proj(fore, nrm))
    radial = _proj(H[:, 5] - H[:, 17], nrm)
    e2 = _unit(radial - (radial * e1).sum(-1, keepdims=True) * e1)
    hp = _proj(hand_dir, nrm)
    add('wrist radial(+)/ulnar(-) deviation', np.degrees(np.arctan2((hp * e2).sum(-1), (hp * e1).sum(-1))),
        [ELBOW[side]] + g(0, 9) + frame_lms)
    fa = _unit(fore)
    add('forearm rotation (0 palm down, 180 palm up)', _angle(_proj(nrm, fa), _proj(np.broadcast_to(-UP, nrm.shape), fa)),
        [ELBOW[side]] + frame_lms)
    return A, L


def all_hand_angles(d3):
    """Both hands: ({name: values}, {name: landmarks})."""
    A, L = {}, {}
    for side in ('R', 'L'):
        a, lm = hand_angles(d3, side)
        A.update(a)
        L.update(lm)
    return A, L

