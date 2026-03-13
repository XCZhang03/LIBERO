import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from collections import defaultdict

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import imageio  # type: ignore
except Exception as e:  # pragma: no cover
    imageio = None  # type: ignore

REQUIRED_CAMERAS = ['agentview', 'birdview', 'robot0_eye_in_hand', 'sideview']

def ensure_deps():
    missing = []
    if cv2 is None:
        missing.append("opencv-python (cv2)")
    if imageio is None:
        missing.append("imageio")
    if missing:
        print("Error: missing dependencies: " + ", ".join(missing))
        print("Install with: conda install -c conda-forge opencv imageio  (or pip install opencv-python imageio)")
        sys.exit(1)


def discover_segments(dirpath: str) -> Dict[str, Dict[str, str]]:
    """Discover all videos grouped by suffix.
    
    Returns:
        Dict mapping suffix -> {camera_name: video_path}
        e.g., {'_seg1': {'agentview': 'agentview_seg1.mp4', ...}, '': {'agentview': 'agentview.mp4', ...}}
    """
    segments: Dict[str, Dict[str, str]] = defaultdict(dict)
    required_cameras = REQUIRED_CAMERAS
    
    for fname in os.listdir(dirpath):
        if not fname.lower().endswith(".mp4") or "tmp" in fname.lower():
            continue
        if "merged" in fname.lower():
            continue
        
        stem = fname[:-4]  # Remove .mp4
        
        # Check if stem starts with any required camera name
        for camera in required_cameras:
            if stem.startswith(camera):
                suffix = stem[len(camera):]  # Everything after camera name
                path = os.path.join(dirpath, fname)
                segments[suffix][camera] = path
                break
    
    return segments


def cap_info(paths: List[str]) -> Tuple[List[cv2.VideoCapture], List[Tuple[int, int]], List[float]]:  # type: ignore
    caps: List[cv2.VideoCapture] = []  # type: ignore
    sizes: List[Tuple[int, int]] = []
    fps_list: List[float] = []
    for p in paths:
        if p is None:
            caps.append(None)  # type: ignore
            sizes.append((0, 0))
            fps_list.append(0.0)
            continue
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {p}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        caps.append(cap)
        sizes.append((w, h))
        fps_list.append(fps)
    return caps, sizes, fps_list


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def infer_tile_size_from_sizes(sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not sizes:
        return 256, 256
    min_w = max(64, min(w for w, _ in sizes if w > 0))
    min_h = max(64, min(h for _, h in sizes if h > 0))
    return even(min_w), even(min_h)


def resize_and_pad(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(tile_w / max(1, iw), tile_h / max(1, ih))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # Pad to tile size centered
    top = (tile_h - nh) // 2
    bottom = tile_h - nh - top
    left = (tile_w - nw) // 2
    right = tile_w - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def merge_four_to_grid(inputs: List[str], output: str, prefer_fps: Optional[float] = None) -> None:
    if len(inputs) != 4:
        raise ValueError("Exactly 4 input videos are required")

    caps, sizes, fps_list = cap_info(inputs)
    try:
        tile_w, tile_h = infer_tile_size_from_sizes(sizes)
        out_h, out_w = tile_h * 2, tile_w * 2
        fps = min([f for f in fps_list if f > 1.0]) if prefer_fps is None else prefer_fps

        # Use imageio writer; try libx264 first, then fallback to mpeg4
        writer = None
        try:
            writer = imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1, quality=8)
        except Exception:
            writer = imageio.get_writer(output, fps=fps, codec="mpeg4", macro_block_size=1, quality=8)
        frame_idx = 0
        with writer as w:
            while True:
                frames: List[np.ndarray] = []
                ok_all = True
                for cap in caps:
                    # if cap is None:
                    #     assert sum([1 for cap in caps if cap is None]) == 1 # only one can be None
                    #     # Create black frame
                    #     frames.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                    #     continue
                    ok, frame = cap.read()
                    if not ok:
                        ok_all = False
                        break
                    frames.append(frame)
                if not ok_all or len(frames) != 4:
                    break

                tiles = [resize_and_pad(fr, tile_w, tile_h) for fr in frames]
                top = np.hstack((tiles[0], tiles[1]))
                bot = np.hstack((tiles[2], tiles[3]))
                grid_bgr = np.vstack((top, bot))

                # Convert BGR (cv2) -> RGB (imageio expects RGB)
                grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
                # Ensure exact size (even dims)
                grid_rgb = cv2.resize(grid_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
                w.append_data(grid_rgb)

                frame_idx += 1
                if frame_idx % 200 == 0:
                    print(f"  wrote {frame_idx} frames -> {output}")
        # print(f"[ok] {output} (frames={frame_idx}, size={out_h}x{out_w}, fps={fps:.2f})")
    finally:
        for cap in caps:
            if cap is not None:
                cap.release()


def merge_in_directory(dirpath: str, overwrite: bool) -> None:
    """Merge all videos in a directory grouped by suffix.
    
    For each suffix, merge agentview, birdview, frontview, sideview
    into merged{suffix}.mp4
    """
    segments = discover_segments(dirpath)
    
    if not segments:
        return
    
    # Required camera order
    required_cameras = REQUIRED_CAMERAS
    
    for suffix in sorted(segments.keys()):
        cameras = segments[suffix]
        
        # Check if all 4 required cameras exist
        if not all(cam in cameras for cam in required_cameras):
            # missing = [cam for cam in required_cameras if cam not in cameras]
            # if (not missing == ['robot0_eye_in_hand']) or (not "pose" in suffix):  # Special case skip message
            #     print(f"[skip] Suffix '{suffix}' in {dirpath} missing cameras: {missing}")
            #     continue
            raise RuntimeError(f"Suffix '{suffix}' in {dirpath} missing required cameras.")
        
        # Prepare inputs in the specified order
        inputs = [cameras[cam] for cam in required_cameras]
        
        # Output name: merged{suffix}.mp4
        output = os.path.join(dirpath, f"merged{suffix}.mp4")
        
        if not overwrite and os.path.exists(output):
            print(f"[skip] Exists: {output}")
        else:
            # print(f"[merge] {dirpath} -> merged{suffix}.mp4 | cameras={required_cameras}")
            merge_four_to_grid(inputs, output)


def discover_eligible_directories(root: str) -> List[str]:
    """Return list of directories that have at least one complete segment.
    
    A complete segment has all 4 camera views: agentview, birdview, frontview, sideview.
    """
    dirs: List[str] = []
    required_cameras = REQUIRED_CAMERAS
    
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith('.'):
            continue
        
        segments = discover_segments(dirpath)
        
        # Check if any segment has all 4 cameras
        has_complete_segment = any(
            all(cam in cameras for cam in required_cameras)
            for cameras in segments.values()
        )
        
        if has_complete_segment:
            dirs.append(dirpath)
    
    return sorted(dirs)


def _worker_process(args_tuple):
    dirpath, overwrite, opencv_threads = args_tuple
    # Reduce oversubscription when using multiprocessing
    if cv2 is not None and hasattr(cv2, 'setNumThreads') and opencv_threads > 0:
        try:
            cv2.setNumThreads(opencv_threads)
        except Exception:
            pass
    try:
        merge_in_directory(dirpath, overwrite)
    except Exception as e:
        print(f"[error] {dirpath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merge 4 camera segment videos into 2x2 grids.")
    parser.add_argument("--root", type=str, default="datasets", help="Root folder containing dataset subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing merged outputs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (0 or -1 for cpu_count)")
    parser.add_argument("--opencv-threads", type=int, default=1, help="OpenCV threads per worker")
    args = parser.parse_args()
    # args.root = "/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/robomimic_dataset/robomimic/datasets_std_0.1_64_chunk40_len200"

    ensure_deps()

    root = args.root
    if not os.path.isdir(root):
        print(f"Error: root directory not found: {root}")
        sys.exit(1)

    print(f"Scanning for segment videos under: {os.path.abspath(root)}")
    eligible_dirs = discover_eligible_directories(root)
    if not eligible_dirs:
        print("No eligible directories with complete segment sets were found.")
        print("Done", 0)
        return

    print(f"Found {len(eligible_dirs)} eligible directories. Starting merges...")
    # Determine worker count
    if args.workers in (0, -1):
        workers = max(1, mp.cpu_count())
    else:
        workers = max(1, args.workers)

    # Prepare tasks
    tasks = [
        (d, args.overwrite, max(0, args.opencv_threads))
        for d in eligible_dirs
    ]

    if workers == 1:
        for t in tasks:
            _worker_process(t)
    else:
        with mp.Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(_worker_process, tasks, chunksize=1):
                pass

    print("Done", len(eligible_dirs))

if __name__ == "__main__":
    main()
