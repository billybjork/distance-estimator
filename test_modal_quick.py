"""
Quick test of Modal deployments with sample video.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.frame_extractor import extract_frames_fast


def main():
    # Extract just 5 frames for quick test
    video_path = Path(__file__).parent / "samples" / "Bill_1.mp4"
    print(f"Extracting frames from {video_path}...")

    frames = extract_frames_fast(str(video_path), target_frames=5)
    print(f"Extracted {len(frames.frame_paths)} frames ({frames.width}x{frames.height})")

    frame_bytes = frames.load_as_bytes()

    # Test SAM2
    print("\n--- Testing SAM2 ---")
    import modal

    try:
        SAM2Inference = modal.Cls.from_name("snowboard-sam2", "SAM2Inference")
        sam2 = SAM2Inference()

        # Use center of first frame as prompt
        center = (frames.width // 2, frames.height // 2)
        print(f"Sending {len(frame_bytes)} frames with prompt point {center}...")

        result = sam2.segment_video.remote(
            frame_data_list=frame_bytes,
            prompt_frame_idx=0,
            prompt_points=[center],
            prompt_labels=[1],
        )

        print(f"SAM2 SUCCESS: Received {len(result['masks'])} masks")
        for i, mask in enumerate(result["masks"]):
            print(f"  Frame {i}: {mask.sum()} pixels masked")

    except Exception as e:
        print(f"SAM2 ERROR: {e}")

    # Test MASt3R
    print("\n--- Testing MASt3R ---")
    try:
        MASt3RInference = modal.Cls.from_name("snowboard-mast3r", "MASt3RInference")
        mast3r = MASt3RInference()

        print(f"Sending {len(frame_bytes)} frames for 3D reconstruction...")
        print("(This may take a few minutes...)")

        result = mast3r.reconstruct.remote(
            image_data_list=frame_bytes,
            image_size=512,
            scene_graph="swin-2",  # Smaller window for fewer frames
            niter1=200,  # Fewer iterations for quick test
            niter2=100,
        )

        print(f"MASt3R SUCCESS: Received {len(result['pointmaps'])} pointmaps")
        for i, pm in enumerate(result["pointmaps"]):
            print(f"  Frame {i}: shape={pm.shape}, range=[{pm.min():.2f}, {pm.max():.2f}]")

    except Exception as e:
        print(f"MASt3R ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    main()
