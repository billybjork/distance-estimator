"""
Full end-to-end analysis test (non-interactive).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pipeline.frame_extractor import extract_frames_fast
from pipeline.geometry import analyze_trick


def main():
    # Extract frames
    video_path = Path(__file__).parent / "samples" / "Bill_1.mp4"
    print(f"Extracting frames from {video_path}...")

    frames = extract_frames_fast(str(video_path), target_frames=10)
    print(f"Extracted {len(frames.frame_paths)} frames ({frames.width}x{frames.height})")

    frame_bytes = frames.load_as_bytes()

    # Run SAM2
    print("\n--- SAM2 Segmentation ---")
    import modal

    SAM2Inference = modal.Cls.from_name("snowboard-sam2", "SAM2Inference")
    sam2 = SAM2Inference()

    center = (frames.width // 2, frames.height // 2)
    print(f"Using center point {center} as prompt...")

    sam2_result = sam2.segment_video.remote(
        frame_data_list=frame_bytes,
        prompt_frame_idx=0,
        prompt_points=[center],
        prompt_labels=[1],
    )
    print(f"Received {len(sam2_result['masks'])} masks")

    # Run MASt3R
    print("\n--- MASt3R 3D Reconstruction ---")
    MASt3RInference = modal.Cls.from_name("snowboard-mast3r", "MASt3RInference")
    mast3r = MASt3RInference()

    print("Running 3D reconstruction (this takes a few minutes)...")
    mast3r_result = mast3r.reconstruct.remote(
        image_data_list=frame_bytes,
        image_size=512,
        scene_graph="swin-4",
    )
    print(f"Received {len(mast3r_result['pointmaps'])} pointmaps")

    # Run analysis
    print("\n--- Trick Analysis ---")
    timestamps = np.array(frames.get_timestamps())
    frame_indices = np.array(frames.frame_indices)

    analysis = analyze_trick(
        masks=sam2_result["masks"],
        pointmaps=mast3r_result["pointmaps"],
        confidences=mast3r_result["confidences"],
        timestamps=timestamps,
        frame_indices=frame_indices,
    )

    print(f"Trajectory: {len(analysis.trajectory_3d)} points")
    print(f"Valid points: {(~np.isnan(analysis.trajectory_3d).any(axis=1)).sum()}")
    print(f"Ground plane normal: [{analysis.ground_plane[0]:.3f}, {analysis.ground_plane[1]:.3f}, {analysis.ground_plane[2]:.3f}]")
    print(f"Board length (3D): {analysis.board_length_3d:.3f} units")
    print(f"Predicted trick segment: frames {analysis.predicted_in} to {analysis.predicted_out}")

    # Compute metrics
    metrics = analysis.compute_metrics(board_length_cm=155.0)

    print("\n" + "=" * 50)
    print("RESULTS (assuming 155cm board)")
    print("=" * 50)
    print(f"Horizontal distance: {metrics['horizontal_m']:.2f} m")
    print(f"Arc length (3D):     {metrics['arc_length_m']:.2f} m")
    print(f"Peak height:         {metrics['peak_height_m']:.2f} m")
    print(f"Airtime:             {metrics['airtime_s']:.2f} s")
    print(f"Takeoff angle:       {metrics['takeoff_angle_deg']:.1f}°")
    print(f"Landing angle:       {metrics['landing_angle_deg']:.1f}°")

    # Save results
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "analysis.npz",
        trajectory_3d=analysis.trajectory_3d,
        heights=analysis.heights,
        timestamps=timestamps,
        frame_indices=frame_indices,
        ground_plane=analysis.ground_plane,
        board_length_3d=analysis.board_length_3d,
    )
    print(f"\nSaved to {output_dir / 'analysis.npz'}")

    print("\n--- DONE ---")


if __name__ == "__main__":
    main()
