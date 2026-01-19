"""
SAM2 video segmentation inference on Modal.

Takes video frames and initial prompts, returns segmentation masks for all frames.
"""

import modal

# Create Modal image with SAM2 dependencies
sam2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy<2",
        "pillow",
        "scipy",
        "tqdm",
        "matplotlib",
        "opencv-python-headless",
        "huggingface_hub",
    )
    # Install SAM2
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && pip install -e .",
    )
    # Download model checkpoint
    .run_commands(
        "mkdir -p /opt/sam2/checkpoints",
        "cd /opt/sam2/checkpoints && "
        "wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    )
)

app = modal.App("snowboard-sam2")


@app.cls(
    image=sam2_image,
    gpu="A10G",  # SAM2 is lighter than MASt3R, A10G is sufficient
    timeout=600,
)
class SAM2Inference:
    """SAM2 video segmentation inference class."""

    @modal.enter()
    def load_model(self):
        """Load SAM2 model on container startup."""
        import sys
        sys.path.insert(0, "/opt/sam2")

        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 model on {self.device}...")

        # Model config and checkpoint
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint = "/opt/sam2/checkpoints/sam2.1_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint,
            device=self.device,
        )
        print("SAM2 model loaded successfully")

    @modal.method()
    def segment_video(
        self,
        frame_data_list: list[bytes],
        prompt_frame_idx: int = 0,
        prompt_points: list[tuple[float, float]] = None,
        prompt_labels: list[int] = None,
        prompt_box: tuple[float, float, float, float] = None,
    ) -> dict:
        """
        Segment an object across video frames.

        Args:
            frame_data_list: List of JPEG image bytes (one per frame)
            prompt_frame_idx: Frame index where prompts are provided
            prompt_points: List of (x, y) point coordinates for prompting
            prompt_labels: List of labels (1=positive/foreground, 0=negative/background)
            prompt_box: Optional bounding box (x1, y1, x2, y2) for initial prompt

        Returns:
            dict with:
                - masks: List of H×W boolean numpy arrays (one per frame)
                - scores: List of confidence scores per frame
                - frame_indices: List of frame indices
        """
        import sys
        sys.path.insert(0, "/opt/sam2")

        import io
        import tempfile
        import os
        import numpy as np
        import torch
        from PIL import Image

        # Save frames to temp directory as JPEGs (SAM2 requirement)
        temp_dir = tempfile.mkdtemp()
        frame_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frame_dir)

        frame_shapes = []
        for i, frame_bytes in enumerate(frame_data_list):
            img = Image.open(io.BytesIO(frame_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            frame_shapes.append((img.height, img.width))
            # SAM2 expects frames named as sequential numbers
            path = os.path.join(frame_dir, f"{i:05d}.jpg")
            img.save(path, "JPEG", quality=95)

        print(f"Processing {len(frame_data_list)} frames...")

        # Initialize inference state
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=frame_dir)

            # Add prompts
            obj_id = 1  # Single object (the rider)

            if prompt_points is not None and prompt_labels is not None:
                points = np.array(prompt_points, dtype=np.float32)
                labels = np.array(prompt_labels, dtype=np.int32)

                _, _, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            elif prompt_box is not None:
                box = np.array(prompt_box, dtype=np.float32)

                _, _, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
            else:
                raise ValueError("Must provide either prompt_points+prompt_labels or prompt_box")

            # Propagate through entire video
            masks = []
            scores = []
            frame_indices = []

            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
                # Convert logits to binary mask
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                masks.append(mask.astype(np.uint8))
                frame_indices.append(frame_idx)
                # Score based on logit magnitude (higher = more confident)
                scores.append(float(mask_logits[0].max().cpu()))

        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Reset predictor state
        self.predictor.reset_state(inference_state)

        return {
            "masks": masks,
            "scores": scores,
            "frame_indices": frame_indices,
            "frame_shapes": frame_shapes,
        }

    @modal.method()
    def segment_with_auto_detect(
        self,
        frame_data_list: list[bytes],
        detection_frame_idx: int = 0,
    ) -> dict:
        """
        Segment the most prominent moving object (assumed to be the rider).

        Uses center-biased detection: assumes rider is near center of frame.

        Args:
            frame_data_list: List of JPEG image bytes
            detection_frame_idx: Frame to use for initial detection

        Returns:
            Same as segment_video()
        """
        import io
        from PIL import Image

        # Get center point of the detection frame
        img = Image.open(io.BytesIO(frame_data_list[detection_frame_idx]))
        center_x = img.width // 2
        center_y = img.height // 2

        # Use center point as positive prompt
        # This assumes the rider is roughly centered in frame
        return self.segment_video(
            frame_data_list=frame_data_list,
            prompt_frame_idx=detection_frame_idx,
            prompt_points=[(center_x, center_y)],
            prompt_labels=[1],  # Positive point
        )


@app.local_entrypoint()
def test_inference():
    """Test SAM2 inference with synthetic frames."""
    import io
    from PIL import Image, ImageDraw

    print("Creating test frames...")

    # Create simple test frames with a moving circle (simulated rider)
    test_frames = []
    for i in range(5):
        img = Image.new("RGB", (640, 480), color=(200, 220, 255))  # Sky blue background
        draw = ImageDraw.Draw(img)

        # Draw "ground" (snow)
        draw.rectangle([0, 350, 640, 480], fill=(250, 250, 250))

        # Draw moving "rider" (circle)
        x = 200 + i * 50
        y = 300
        draw.ellipse([x - 30, y - 50, x + 30, y + 50], fill=(50, 50, 150))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        test_frames.append(buf.getvalue())

    print(f"Testing with {len(test_frames)} frames...")

    # Run inference
    sam2 = SAM2Inference()

    # Test with explicit point prompt (on the "rider" in first frame)
    result = sam2.segment_video.remote(
        frame_data_list=test_frames,
        prompt_frame_idx=0,
        prompt_points=[(200, 300)],  # Center of rider in first frame
        prompt_labels=[1],
    )

    print(f"Received {len(result['masks'])} masks")
    for i, mask in enumerate(result["masks"]):
        print(f"  Frame {i}: shape={mask.shape}, pixels={mask.sum()}")

    print("Test completed successfully!")
