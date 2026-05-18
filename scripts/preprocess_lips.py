"""
Extract frames from GRID MPG videos and crop the lip region.

Pipeline per video:
  1. ffmpeg extracts JPG frames at 25 fps  →  raw_frames/
  2. face_alignment detects 68 landmarks per frame
  3. Similarity transform aligns face to canonical front view
  4. Lip region is cropped and saved as numbered JPGs  →  lip/

Usage (inside the Apptainer container):
  python scripts/preprocess_lips.py \
      --raw_dir   /data/grid/raw_videos \
      --out_dir   /data/grid/lip \
      --n_workers 8

Requirements (install once inside the container):
  pip install --user face-alignment
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool, current_process

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Face geometry helpers (same logic as demo.py / extract_lip.py)
# ---------------------------------------------------------------------------

def _get_ref_landmarks(size: int = 256, padding: float = 0.25) -> np.ndarray:
    """Canonical 51-point face template (landmarks 17-67)."""
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
         0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
         0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
         0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
         0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
         0.551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
         0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
         0.490127, 0.42689]
    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
         0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
         0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
         0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
         0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
         0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
         0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
         0.831803, 0.824182]
    x = np.array(x)
    y = np.array(y)
    x = (x + padding) / (2 * padding + 1) * size
    y = (y + padding) / (2 * padding + 1) * size
    return np.stack([x, y], axis=1)


REF_LANDMARKS = _get_ref_landmarks(256)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 2D similarity transform matrix (scale + rotation + translation)."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    c1, c2 = src.mean(0), dst.mean(0)
    src -= c1
    dst -= c2
    s1, s2 = src.std(), dst.std()
    src /= (s1 + 1e-8)
    dst /= (s2 + 1e-8)
    U, _, Vt = np.linalg.svd(src.T @ dst)
    R = (U @ Vt).T
    M = np.vstack([
        np.hstack([(s2 / (s1 + 1e-8)) * R, (c2 - (s2 / (s1 + 1e-8)) * R @ c1).reshape(2, 1)]),
        [0., 0., 1.]
    ])
    return M


def _crop_lip(frame: np.ndarray, landmarks: np.ndarray,
              crop_w: int = 128, crop_h: int = 64) -> np.ndarray | None:
    """Align face and crop lip region. Returns None if transform fails."""
    shape = landmarks[17:]                          # 51 points (same as ref)
    try:
        M = _similarity_transform(shape, REF_LANDMARKS)
    except Exception:
        return None
    aligned = cv2.warpAffine(frame, M[:2], (256, 256))
    # Lip centre is the mean of the last 20 reference points
    cx, cy = REF_LANDMARKS[-20:].mean(0).astype(int)
    half_w = crop_w // 2
    half_h = crop_h // 2
    lip = aligned[cy - half_h: cy + half_h, cx - crop_w: cx + crop_w]
    if lip.shape[0] != crop_h or lip.shape[1] != crop_w * 2:
        # Fall back to resize if crop went out of bounds
        lip = cv2.resize(aligned[max(0, cy - half_h): cy + half_h,
                                  max(0, cx - crop_w): cx + crop_w],
                         (crop_w * 2, crop_h))  # NOTE: actual output is 128×64
    # Final resize to exactly (128, 64) — width × height
    return cv2.resize(lip, (128, 64), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# Per-video worker
# ---------------------------------------------------------------------------

def _process_video(args: tuple) -> tuple[str, str]:
    """
    Extract frames from one MPG video, detect landmarks, crop lips.
    Returns (video_rel_path, status_message).
    """
    video_path, out_dir, device = args

    # e.g.  raw_videos/s14/video/mpg_6000/srwt9p.mpg
    #  ->   lip/s14/video/mpg_6000/srwt9p/
    rel = os.path.relpath(video_path, os.path.dirname(os.path.dirname(out_dir)))
    rel_noext = os.path.splitext(rel)[0]
    save_dir = os.path.join(out_dir, rel_noext)

    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        return (rel_noext, 'SKIP (already processed)')

    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: extract frames with ffmpeg ---
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ['ffmpeg', '-y', '-i', video_path,
               '-qscale:v', '2', '-r', '25',
               os.path.join(tmp, '%d.jpg')]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return (rel_noext, f'FAIL ffmpeg: {result.stderr.decode()[:200]}')

        frame_files = sorted(
            [f for f in os.listdir(tmp) if f.endswith('.jpg')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        if not frame_files:
            return (rel_noext, 'FAIL: no frames extracted')

        frames = [cv2.imread(os.path.join(tmp, f)) for f in frame_files]

        # --- Step 2: detect landmarks (lazy import to avoid CUDA init in main) ---
        try:
            import face_alignment  # noqa: PLC0415
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=device,
            )
        except ImportError:
            return (rel_noext, 'FAIL: face_alignment not installed. '
                    'Run: pip install --user face-alignment')

        saved = 0
        for idx, (frame, fname) in enumerate(zip(frames, frame_files)):
            preds = fa.get_landmarks(frame)
            if preds is None:
                continue
            # Pick the largest face if multiple detected
            landmarks = preds[0]
            lip = _crop_lip(frame, landmarks)
            if lip is None:
                continue
            cv2.imwrite(os.path.join(save_dir, f'{idx + 1}.jpg'), lip)
            saved += 1

    if saved == 0:
        return (rel_noext, 'FAIL: no lips saved (no faces detected)')
    return (rel_noext, f'OK ({saved} frames)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import glob

    # Collect all MPG files
    pattern = os.path.join(args.raw_dir, '**', '*.mpg')
    videos = sorted(glob.glob(pattern, recursive=True))
    if not videos:
        print(f'No .mpg files found under {args.raw_dir}')
        sys.exit(1)

    print(f'Found {len(videos)} videos.')
    print(f'Output dir : {args.out_dir}')
    print(f'Workers    : {args.n_workers}')
    print(f'Device     : {args.device}')
    print()

    tasks = [(v, args.out_dir, args.device) for v in videos]

    if args.n_workers == 1:
        for i, task in enumerate(tasks):
            rel, status = _process_video(task)
            print(f'[{i+1}/{len(tasks)}] {rel}: {status}')
    else:
        # Note: face_alignment initialises CUDA per-worker — keep n_workers low
        # when using GPU (1 per GPU); use CPU with higher n_workers.
        with Pool(processes=args.n_workers) as pool:
            for i, (rel, status) in enumerate(pool.imap_unordered(_process_video, tasks)):
                print(f'[{i+1}/{len(tasks)}] {rel}: {status}', flush=True)

    print('\nPreprocessing complete.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', required=True,
                   help='Root dir containing raw MPG videos (e.g. /data/grid/raw_videos)')
    p.add_argument('--out_dir', required=True,
                   help='Output dir for lip crops (e.g. /data/grid/lip)')
    p.add_argument('--n_workers', type=int, default=4,
                   help='Parallel workers. Use 1 with --device cuda to avoid CUDA conflicts.')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                   help='Device for face_alignment landmark detector')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
