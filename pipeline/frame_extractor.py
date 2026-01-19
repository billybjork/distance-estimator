"""
Local frame extraction from video files.

Uses ffmpeg to extract frames at specified intervals.
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class ExtractedFrames:
    """Container for extracted video frames."""
    frame_paths: List[Path]
    frame_indices: List[int]
    fps: float
    total_frames: int
    width: int
    height: int

    def get_timestamps(self) -> list[float]:
        """Get timestamp in seconds for each extracted frame."""
        return [idx / self.fps for idx in self.frame_indices]

    def load_as_bytes(self) -> list[bytes]:
        """Load all frames as bytes (for sending to Modal)."""
        frame_bytes = []
        for path in self.frame_paths:
            with open(path, "rb") as f:
                frame_bytes.append(f.read())
        return frame_bytes


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    import json
    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError("No video stream found")

    # Parse frame rate (can be "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0)),
        "total_frames": int(video_stream.get("nb_frames", 0)) or int(fps * float(data.get("format", {}).get("duration", 0))),
    }


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path = None,
    target_frames: int = 20,
    max_dimension: int = 720,
) -> ExtractedFrames:
    """
    Extract frames from video using linear sampling.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames (uses temp dir if None)
        target_frames: Number of frames to extract
        max_dimension: Resize frames so max dimension is this size

    Returns:
        ExtractedFrames object with paths and metadata
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video info
    info = get_video_info(video_path)
    total_frames = info["total_frames"]
    fps = info["fps"]

    # Calculate frame indices (linear sampling)
    import numpy as np
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate scale filter
    w, h = info["width"], info["height"]
    if max(w, h) > max_dimension:
        if w > h:
            scale = f"scale={max_dimension}:-2"
        else:
            scale = f"scale=-2:{max_dimension}"
    else:
        scale = None

    # Extract frames using ffmpeg
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        timestamp = frame_idx / fps
        output_path = output_dir / f"frame_{i:04d}.jpg"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",  # High quality JPEG
        ]

        if scale:
            cmd.extend(["-vf", scale])

        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to extract frame {frame_idx}: {result.stderr}")
            continue

        frame_paths.append(output_path)

    # Get actual output dimensions from first frame
    if frame_paths:
        from PIL import Image
        with Image.open(frame_paths[0]) as img:
            out_w, out_h = img.size
    else:
        out_w, out_h = info["width"], info["height"]

    return ExtractedFrames(
        frame_paths=frame_paths,
        frame_indices=frame_indices[:len(frame_paths)],
        fps=fps,
        total_frames=total_frames,
        width=out_w,
        height=out_h,
    )


def extract_frames_fast(
    video_path: str | Path,
    output_dir: str | Path = None,
    target_frames: int = 20,
    max_dimension: int = 720,
) -> ExtractedFrames:
    """
    Extract frames using a single ffmpeg call with frame selection filter.
    Faster than extract_frames() for many frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    total_frames = info["total_frames"]
    fps = info["fps"]

    import numpy as np
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build select filter for specific frames
    select_expr = "+".join([f"eq(n,{idx})" for idx in frame_indices])

    # Build filter chain
    filters = [f"select='{select_expr}'"]

    w, h = info["width"], info["height"]
    if max(w, h) > max_dimension:
        if w > h:
            filters.append(f"scale={max_dimension}:-2")
        else:
            filters.append(f"scale=-2:{max_dimension}")

    filter_str = ",".join(filters)

    # Single ffmpeg call to extract all frames
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", filter_str,
        "-vsync", "vfr",
        "-q:v", "2",
        str(output_dir / "frame_%04d.jpg"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    # Collect output paths
    frame_paths = sorted(output_dir.glob("frame_*.jpg"))

    # Get output dimensions
    if frame_paths:
        from PIL import Image
        with Image.open(frame_paths[0]) as img:
            out_w, out_h = img.size
    else:
        out_w, out_h = info["width"], info["height"]

    return ExtractedFrames(
        frame_paths=frame_paths,
        frame_indices=frame_indices[:len(frame_paths)],
        fps=fps,
        total_frames=total_frames,
        width=out_w,
        height=out_h,
    )


if __name__ == "__main__":
    # Test with sample video
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test path
        video_path = "../samples/Bill_1.mp4"

    print(f"Extracting frames from: {video_path}")
    frames = extract_frames_fast(video_path, target_frames=10)

    print(f"Extracted {len(frames.frame_paths)} frames")
    print(f"  Video: {frames.total_frames} frames @ {frames.fps:.2f} fps")
    print(f"  Output size: {frames.width}x{frames.height}")
    print(f"  Frame indices: {frames.frame_indices}")
    print(f"  Timestamps: {[f'{t:.2f}s' for t in frames.get_timestamps()]}")
    print(f"  Saved to: {frames.frame_paths[0].parent}")
