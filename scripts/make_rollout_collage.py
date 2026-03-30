from __future__ import annotations

import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


TARGET_DIR = Path('/workspace/le-wm/runs/20260328-132542-pusht')
OUTPUT_PATH = TARGET_DIR / 'rollout_collage.mp4'
GLOB_PATTERN = 'rollout_*.mp4'
GRID_ROWS = 5
GRID_COLS = 5
TILE_WIDTH = 160
TILE_HEIGHT = 90
INTRO_VIDEO_COUNT = 10
INTRO_FRAMES_PER_VIDEO = 10
FPS_OVERRIDE = None


_DIGIT_RE = re.compile(r'(\d+)')


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in _DIGIT_RE.split(path.name)]


def list_rollout_videos(target_dir: Path) -> list[Path]:
    videos = sorted(target_dir.glob(GLOB_PATTERN), key=natural_sort_key)
    if not videos:
        raise FileNotFoundError(f'No rollout videos found under {target_dir} matching {GLOB_PATTERN}')
    return videos


def read_video_frames(path: Path, tile_size: tuple[int, int]) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    tile_w, tile_h = tile_size
    frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        half_height = max(1, frame.shape[0] // 2)
        frame = frame[:half_height]
        frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f'Video contained no readable frames: {path}')
    return frames, fps


def make_intro_frames(clips: list[list[np.ndarray]], canvas_size: tuple[int, int]) -> list[np.ndarray]:
    width, height = canvas_size
    intro_frames: list[np.ndarray] = []
    for clip in clips[:INTRO_VIDEO_COUNT]:
        if not clip:
            continue
        limit = min(INTRO_FRAMES_PER_VIDEO, len(clip))
        for idx in range(limit):
            frame = cv2.resize(clip[idx], (width, height), interpolation=cv2.INTER_LINEAR)
            intro_frames.append(frame)
    return intro_frames


def make_grid_frames(clips: list[list[np.ndarray]], rows: int, cols: int, tile_size: tuple[int, int]) -> list[np.ndarray]:
    tile_w, tile_h = tile_size
    grid_slots = rows * cols
    chosen = clips[:grid_slots]
    if len(chosen) < grid_slots:
        chosen = chosen + [[] for _ in range(grid_slots - len(chosen))]

    max_len = max((len(clip) for clip in chosen), default=0)
    if max_len == 0:
        return []

    grid_frames: list[np.ndarray] = []
    black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    for frame_idx in range(max_len):
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for slot_idx, clip in enumerate(chosen):
            row = slot_idx // cols
            col = slot_idx % cols
            y0 = row * tile_h
            x0 = col * tile_w
            frame = clip[frame_idx] if frame_idx < len(clip) else black
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = frame
        grid_frames.append(canvas)
    return grid_frames


def main() -> None:
    videos = list_rollout_videos(TARGET_DIR)
    clips: list[list[np.ndarray]] = []
    fps = FPS_OVERRIDE

    for video in videos:
        frames, video_fps = read_video_frames(video, tile_size=(TILE_WIDTH, TILE_HEIGHT))
        clips.append(frames)
        if fps is None:
            fps = video_fps

    if fps is None:
        fps = 15.0

    canvas_size = (GRID_COLS * TILE_WIDTH, GRID_ROWS * TILE_HEIGHT)
    intro_frames = make_intro_frames(clips, canvas_size=canvas_size)
    grid_frames = make_grid_frames(clips, rows=GRID_ROWS, cols=GRID_COLS, tile_size=(TILE_WIDTH, TILE_HEIGHT))
    all_frames = intro_frames + grid_frames
    if not all_frames:
        raise RuntimeError('No output frames were generated.')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUTPUT_PATH, fps=fps, codec='libx264') as writer:
        for frame in all_frames:
            writer.append_data(frame)

    print(f'Wrote collage video to {OUTPUT_PATH}')
    print(f'Used {min(len(clips), GRID_ROWS * GRID_COLS)} videos in the {GRID_ROWS}x{GRID_COLS} grid.')


if __name__ == '__main__':
    main()
