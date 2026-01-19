"""
MASt3R inference on Modal.

Takes a set of images and returns dense 3D pointmaps aligned in a common world coordinate system.
"""

import modal

# Create Modal image with MASt3R dependencies
mast3r_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "numpy<2",
        "pillow",
        "scipy",
        "tqdm",
        "matplotlib",
        "opencv-python-headless",
        "trimesh",
        "pyglet<2",
        "huggingface_hub",
        "gradio",  # needed for some utils
        "roma",
        "einops",
    )
    # Clone MASt3R with submodules and install requirements
    .run_commands(
        "git clone --recursive https://github.com/naver/mast3r /opt/mast3r",
        "cd /opt/mast3r && pip install -r requirements.txt",
        "cd /opt/mast3r && pip install -r dust3r/requirements.txt",
        # Force numpy<2 for torchvision compatibility
        "pip install 'numpy<2'",
    )
    # Download model checkpoint
    .run_commands(
        "mkdir -p /opt/mast3r/checkpoints",
        "wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -O /opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    # Add mast3r to Python path
    .env({"PYTHONPATH": "/opt/mast3r:/opt/mast3r/dust3r"})
)

app = modal.App("snowboard-mast3r")

# Create a volume to cache model weights between runs
model_cache = modal.Volume.from_name("mast3r-cache", create_if_missing=True)


@app.cls(
    image=mast3r_image,
    gpu="A100",  # MASt3R needs significant VRAM
    timeout=600,
    volumes={"/cache": model_cache},
)
class MASt3RInference:
    """MASt3R inference class for 3D reconstruction from multiple images."""

    @modal.enter()
    def load_model(self):
        """Load MASt3R model on container startup."""
        import sys
        sys.path.insert(0, "/opt/mast3r")

        import torch
        from mast3r.model import AsymmetricMASt3R

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MASt3R model on {self.device}...")

        # Load from local checkpoint
        checkpoint_path = "/opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        self.model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(self.device)
        # Note: Don't call model.eval() here as sparse_global_alignment needs
        # the model to produce outputs with gradients for optimization
        print("Model loaded successfully")

    @modal.method()
    def reconstruct(
        self,
        image_data_list: list[bytes],
        image_size: int = 512,
        scene_graph: str = "swin-4",  # sliding window of 4 for video frames
        lr1: float = 0.07,
        niter1: int = 500,
        lr2: float = 0.014,
        niter2: int = 200,
        min_conf_thr: float = 1.5,
    ) -> dict:
        """
        Reconstruct 3D scene from multiple images.

        Args:
            image_data_list: List of image bytes (PNG/JPEG)
            image_size: Resize images to this size (default 512)
            scene_graph: Pairing strategy - 'complete', 'swin-N', 'oneref'
            lr1, niter1: Learning rate and iterations for coarse alignment
            lr2, niter2: Learning rate and iterations for fine alignment
            min_conf_thr: Minimum confidence threshold for points

        Returns:
            dict with:
                - pointmaps: List of H×W×3 numpy arrays (one per image)
                - confidences: List of H×W confidence arrays
                - poses: List of 4×4 camera-to-world matrices
                - focals: List of focal lengths
                - image_shapes: List of (H, W) tuples
        """
        import sys
        sys.path.insert(0, "/opt/mast3r")

        import io
        import tempfile
        import os
        import numpy as np
        import torch
        from PIL import Image

        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.device import to_numpy

        # Save images to temp files (MASt3R expects file paths)
        temp_dir = tempfile.mkdtemp()
        file_list = []

        for i, img_bytes in enumerate(image_data_list):
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            img.save(path, "JPEG", quality=95)
            file_list.append(path)

        print(f"Processing {len(file_list)} images...")

        # Load images
        imgs = load_images(file_list, size=image_size, verbose=False)

        # Make pairs based on scene graph
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True
        )
        print(f"Created {len(pairs)} image pairs")

        # Cache directory for optimization
        cache_dir = tempfile.mkdtemp()

        # Ensure gradients are enabled for optimization
        torch.set_grad_enabled(True)

        # Run sparse global alignment (requires gradients for optimization)
        scene = sparse_global_alignment(
            file_list,
            pairs,
            cache_dir,
            self.model,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            device=self.device,
            opt_depth=True,
            shared_intrinsics=True,  # Assume same camera for all frames
            matching_conf_thr=2.0,
        )

        # Extract results
        pts3d_list, _, confs_list = to_numpy(scene.get_dense_pts3d(clean_depth=True))
        poses = to_numpy(scene.get_im_poses())
        focals = to_numpy(scene.get_focals())

        # Get image shapes
        image_shapes = [(img['img'].shape[1], img['img'].shape[2]) for img in imgs]

        # Filter by confidence and convert to serializable format
        pointmaps = []
        confidences = []

        for pts, conf in zip(pts3d_list, confs_list):
            pointmaps.append(pts.astype(np.float32))
            confidences.append(conf.astype(np.float32))

        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

        return {
            "pointmaps": pointmaps,
            "confidences": confidences,
            "poses": [p.astype(np.float32) for p in poses],
            "focals": [float(f) for f in focals],
            "image_shapes": image_shapes,
        }


@app.local_entrypoint()
def test_inference():
    """Test the MASt3R inference with sample images."""
    import os
    from pathlib import Path

    # Find sample images (use first 5 frames from a video)
    sample_dir = Path(__file__).parent.parent / "samples"

    # For testing, we'll create simple test images
    print("Creating test images...")
    from PIL import Image
    import io

    test_images = []
    for i in range(3):
        # Create simple gradient images for testing
        img = Image.new("RGB", (640, 480), color=(100 + i * 50, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        test_images.append(buf.getvalue())

    print(f"Testing with {len(test_images)} images...")

    # Run inference
    mast3r = MASt3RInference()
    result = mast3r.reconstruct.remote(test_images)

    print(f"Received {len(result['pointmaps'])} pointmaps")
    for i, pm in enumerate(result["pointmaps"]):
        print(f"  Frame {i}: shape={pm.shape}, range=[{pm.min():.2f}, {pm.max():.2f}]")

    print("Test completed successfully!")
