"""
End-to-end pipeline test.

Extracts frames from a sample video, runs MASt3R and SAM2 on Modal,
and verifies the outputs are usable.
"""

import sys
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.frame_extractor import extract_frames_fast


def test_frame_extraction(video_path: str):
    """Test local frame extraction."""
    print("\n" + "=" * 60)
    print("STEP 1: Frame Extraction")
    print("=" * 60)

    frames = extract_frames_fast(video_path, target_frames=10)

    print(f"  Extracted {len(frames.frame_paths)} frames")
    print(f"  Size: {frames.width}x{frames.height}")
    print(f"  Timestamps: {[f'{t:.2f}s' for t in frames.get_timestamps()]}")

    return frames


def test_sam2(frame_bytes: list[bytes], prompt_point: tuple[int, int]):
    """Test SAM2 inference on Modal."""
    print("\n" + "=" * 60)
    print("STEP 2: SAM2 Segmentation (Modal)")
    print("=" * 60)

    import modal

    # Look up the deployed SAM2 function
    SAM2Inference = modal.Cls.from_name("snowboard-sam2", "SAM2Inference")
    sam2 = SAM2Inference()

    print(f"  Sending {len(frame_bytes)} frames to Modal...")
    print(f"  Prompt point: {prompt_point}")

    result = sam2.segment_video.remote(
        frame_data_list=frame_bytes,
        prompt_frame_idx=0,
        prompt_points=[prompt_point],
        prompt_labels=[1],
    )

    print(f"  Received {len(result['masks'])} masks")
    for i, mask in enumerate(result["masks"]):
        pixels = mask.sum()
        total = mask.shape[0] * mask.shape[1]
        print(f"    Frame {i}: {pixels} pixels ({100*pixels/total:.1f}% of frame)")

    return result


def test_mast3r(frame_bytes: list[bytes]):
    """Test MASt3R inference on Modal."""
    print("\n" + "=" * 60)
    print("STEP 3: MASt3R 3D Reconstruction (Modal)")
    print("=" * 60)

    import modal

    # Look up the deployed MASt3R function
    MASt3RInference = modal.Cls.from_name("snowboard-mast3r", "MASt3RInference")
    mast3r = MASt3RInference()

    print(f"  Sending {len(frame_bytes)} frames to Modal...")
    print("  (This may take a few minutes for global alignment)")

    result = mast3r.reconstruct.remote(
        image_data_list=frame_bytes,
        image_size=512,
        scene_graph="swin-4",  # Sliding window for video
    )

    print(f"  Received {len(result['pointmaps'])} pointmaps")
    for i, pm in enumerate(result["pointmaps"]):
        print(f"    Frame {i}: shape={pm.shape}, range=[{pm.min():.2f}, {pm.max():.2f}]")

    return result


def visualize_results(frames, sam2_result, mast3r_result, output_dir: Path):
    """Create simple visualizations of the results."""
    print("\n" + "=" * 60)
    print("STEP 4: Visualization")
    print("=" * 60)

    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize masks overlaid on frames
    for i, (frame_path, mask) in enumerate(zip(frames.frame_paths, sam2_result["masks"])):
        img = Image.open(frame_path).convert("RGBA")

        # Create mask overlay (red, semi-transparent)
        mask_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_img[mask > 0] = [255, 0, 0, 128]  # Red with 50% opacity
        mask_overlay = Image.fromarray(mask_img, "RGBA")

        # Resize mask to match image if needed
        if mask_overlay.size != img.size:
            mask_overlay = mask_overlay.resize(img.size, Image.NEAREST)

        # Composite
        result = Image.alpha_composite(img, mask_overlay)
        result.save(output_dir / f"masked_frame_{i:04d}.png")

    print(f"  Saved masked frames to {output_dir}")

    # Basic 3D trajectory plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Extract centroids from masked pointmaps
        centroids = []
        for pm, mask in zip(mast3r_result["pointmaps"], sam2_result["masks"]):
            # Resize mask to match pointmap if needed
            if mask.shape != pm.shape[:2]:
                from scipy.ndimage import zoom
                scale_h = pm.shape[0] / mask.shape[0]
                scale_w = pm.shape[1] / mask.shape[1]
                mask_resized = zoom(mask.astype(float), (scale_h, scale_w), order=0) > 0.5
            else:
                mask_resized = mask > 0

            if mask_resized.sum() > 0:
                rider_points = pm[mask_resized]
                centroid = rider_points.mean(axis=0)
                centroids.append(centroid)

        if centroids:
            centroids = np.array(centroids)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            ax.plot(centroids[:, 0], centroids[:, 2], centroids[:, 1], "b-o", linewidth=2)
            ax.scatter(centroids[0, 0], centroids[0, 2], centroids[0, 1], c="g", s=100, label="Start")
            ax.scatter(centroids[-1, 0], centroids[-1, 2], centroids[-1, 1], c="r", s=100, label="End")

            ax.set_xlabel("X")
            ax.set_ylabel("Z (forward)")
            ax.set_zlabel("Y (up)")
            ax.legend()
            ax.set_title("Rider Trajectory (3D)")

            plt.savefig(output_dir / "trajectory_3d.png", dpi=150)
            plt.close()

            print(f"  Saved trajectory plot to {output_dir / 'trajectory_3d.png'}")

    except ImportError:
        print("  Skipping 3D plot (matplotlib not available)")


def main():
    """Run the full pipeline test."""
    # Find sample video
    samples_dir = Path(__file__).parent / "samples"
    videos = list(samples_dir.glob("*.mp4")) + list(samples_dir.glob("*.MOV"))

    if not videos:
        print("No sample videos found in ./samples/")
        print("Please add a video file to test with.")
        sys.exit(1)

    # Use first video
    video_path = videos[0]
    print(f"Using video: {video_path.name}")

    # Step 1: Extract frames
    frames = test_frame_extraction(str(video_path))
    frame_bytes = frames.load_as_bytes()

    # Get user input for prompt point (or use center)
    print(f"\nFrame size: {frames.width}x{frames.height}")
    print("Enter prompt point (x,y) to mark the rider in the first frame,")
    print(f"or press Enter to use center ({frames.width//2}, {frames.height//2}): ", end="")

    try:
        user_input = input().strip()
        if user_input:
            x, y = map(int, user_input.split(","))
            prompt_point = (x, y)
        else:
            prompt_point = (frames.width // 2, frames.height // 2)
    except (ValueError, EOFError):
        prompt_point = (frames.width // 2, frames.height // 2)

    # Step 2: Run SAM2
    sam2_result = test_sam2(frame_bytes, prompt_point)

    # Step 3: Run MASt3R
    mast3r_result = test_mast3r(frame_bytes)

    # Step 4: Visualize
    output_dir = Path(__file__).parent / "test_output"
    visualize_results(frames, sam2_result, mast3r_result, output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
