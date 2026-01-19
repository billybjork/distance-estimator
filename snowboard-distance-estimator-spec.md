# Snowboard Trick Distance Estimator

## Overview

A tool to estimate distance traveled during snowboard tricks (jumps, rails, boxes, jibs) using only video footage. Uses modern ML-based 3D reconstruction to avoid traditional camera calibration pipelines.

**Supported features:**
- **Jumps**: Kickers, step-downs, gaps, natural features — measures air distance
- **Rails/Boxes**: Flat bars, down rails, rainbow rails, boxes — measures grind/slide distance
- **Any feature**: Anything where you want to measure how far the rider traveled

## Core Problem

Estimating metric distance from video requires solving **scale ambiguity** — monocular 3D reconstruction recovers relative structure but not absolute size. We solve this by:

1. Using multi-view reconstruction (multiple frames from video)
2. Anchoring scale via a known reference object (the snowboard, ~150-160cm)

## Technical Approach

### Philosophy

Lean on "bitter lesson" learnings — prefer end-to-end learned systems over hand-engineered pipelines. Let models do the heavy lifting.

### Key Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| 3D Reconstruction | **MASt3R** (Naver Labs, 2024) | Direct image pairs → dense 3D pointmaps. No explicit camera calibration, no COLMAP, no SfM pipeline. Learned geometric priors end-to-end. |
| Object Tracking | **SAM2** | Video segmentation for rider tracking across frames. Well-supported, handles occlusion. |
| Scale Reference | **Snowboard length** | Present in every frame, known size (~150-160cm), rigid object. |

### Why MASt3R over alternatives

- No camera intrinsics needed
- No manual feature matching
- No bundle adjustment tuning
- Handles in-the-wild footage (phone videos, action cameras)
- Outputs dense pointmaps (3D coordinate per pixel) which makes mask→3D trivial

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PROCESSING PIPELINE                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Video → Frame Sampler → [frame_0, frame_1, ..., frame_N]   │
│                                ↓                            │
│                    ┌──────────┴──────────┐                  │
│                    ↓                      ↓                 │
│               MASt3R                    SAM2                │
│           (dense 3D pointmaps)    (rider masks)             │
│                    ↓                      ↓                 │
│                    └──────────┬──────────┘                  │
│                               ↓                             │
│                    Fuse: mask × pointmap → 3D trajectory    │
│                               ↓                             │
│                    RANSAC ground plane detection            │
│                               ↓                             │
│                    Compute world frame (Y-up, Z-forward)    │
│                               ↓                             │
│                    Height series → auto in/out prediction   │
│                               ↓                             │
│                    Board measurement → scale factor         │
│                               ↓                             │
│                         Results object                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Decided Implementation Details

### Frame Sampling

- **Strategy**: Linear sampling across video duration
- **Target**: ~20 frames for MASt3R (works best with 10-30 images)
- **Rationale**: Simple, uniform coverage. Motion-adaptive sampling adds complexity without clear benefit — tricks have fairly continuous motion and displacement is naturally captured.

```python
def sample_frames(video_path, target_frames=20):
    total_frames = get_frame_count(video_path)
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    return extract_frames(video_path, indices), indices
```

### Rider Tracking

- **Strategy**: Track rider centroid via SAM2 mask
- **Rationale**: Simple, robust. Board is part of rider mask so centroid naturally weighted toward it.
- **Spin/flip invariance**: Centroid tracking is inherently invariant to rider rotation. Even during 720s or backflips, the center of mass follows a smooth trajectory. SAM2 tracks the whole rider silhouette, so the centroid remains stable regardless of body orientation.
- **Accuracy target**: 90% is acceptable for this use case

### Mask → 3D Fusion

MASt3R outputs dense pointmaps (H×W×3), so fusion is just array indexing:

```python
mask = sam2_masks[frame_idx]           # H×W boolean
pointmap = mastr_points[frame_idx]     # H×W×3

rider_points_3d = pointmap[mask]       # N×3
position = rider_points_3d.mean(axis=0)  # centroid
```

### Feature Segment Detection (Auto In/Out)

Automatic detection of when rider is "on feature" (in the air or on a rail):

```python
def detect_feature_segment(heights, threshold=0.3):
    """
    Find longest contiguous segment above height threshold.
    Works for both jumps (airborne) and rails (elevated).
    
    Threshold of 0.3m catches rails/boxes while ignoring
    minor terrain undulation.
    """
    on_feature = heights > threshold
    
    changes = np.diff(np.concatenate([[False], on_feature, [False]]).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    if len(starts) == 0:
        return 0, len(heights) - 1  # fallback: use full video
    
    lengths = ends - starts
    best = np.argmax(lengths)
    
    return starts[best], ends[best] - 1
```

**Note**: This heuristic works for most cases but user can always adjust via UI. The height threshold is tunable — lower for low boxes, higher to ignore small terrain features.

### Distance Metrics (all computed)

1. **Horizontal distance**: Start-to-end distance in ground plane (XZ)
2. **Arc length**: Sum of consecutive 3D Euclidean distances (total path traveled)
3. **Peak height**: Maximum height above ground during feature
4. **Takeoff angle**: Trajectory angle relative to ground at in-point (degrees)
5. **Landing angle**: Trajectory angle relative to ground at out-point (degrees)

### Board Length (Scale Reference)

- **User provides manually** (no auto-detection)
- **Default**: 155 cm (common all-mountain length)
- **Input**: Simple number field in UI, user enters their actual board length for accurate results
- **How it's used**: Measure board's apparent length in 3D reconstruction, compute scale factor to correct all distances

---

## Data Structures

### Core Results Object

```python
@dataclass
class TrickAnalysis:
    trajectory_3d: np.ndarray      # N×3 positions in world coords
    heights: np.ndarray            # N floats (height above ground)
    timestamps: np.ndarray         # N floats (seconds from video start)
    frame_indices: np.ndarray      # N ints (original frame numbers)
    predicted_in: int              # auto-detected trick start (index into above arrays)
    predicted_out: int             # auto-detected trick end (index into above arrays)
    ground_plane: np.ndarray       # plane coefficients (a, b, c, d) for ax+by+cz+d=0
    board_length_3d: float         # measured board length in reconstruction (arbitrary units)
    
    def compute_metrics(self, 
                        in_idx: int = None, 
                        out_idx: int = None,
                        board_length_cm: float = 155.0) -> dict:
        """Compute trick metrics with optional user overrides."""
        in_idx = in_idx if in_idx is not None else self.predicted_in
        out_idx = out_idx if out_idx is not None else self.predicted_out
        
        # Scale factor: known board length / measured board length
        scale = (board_length_cm / 100) / self.board_length_3d
        
        segment = self.trajectory_3d[in_idx:out_idx+1] * scale
        heights = self.heights[in_idx:out_idx+1] * scale
        
        # Horizontal distance (start to end, XZ plane)
        horizontal = np.linalg.norm(segment[-1, [0,2]] - segment[0, [0,2]])
        
        # Arc length (sum of consecutive 3D distances)
        deltas = np.diff(segment, axis=0)
        arc_length = np.sum(np.linalg.norm(deltas, axis=1))
        
        # Takeoff angle (angle of initial velocity relative to ground)
        takeoff_vec = segment[1] - segment[0] if len(segment) > 1 else np.array([0,0,1])
        takeoff_angle = self._angle_to_ground(takeoff_vec)
        
        # Landing angle (angle of final velocity relative to ground)
        landing_vec = segment[-1] - segment[-2] if len(segment) > 1 else np.array([0,0,1])
        landing_angle = self._angle_to_ground(landing_vec)
        
        return {
            'horizontal_m': horizontal,
            'arc_length_m': arc_length,
            'peak_height_m': heights.max(),
            'airtime_s': self.timestamps[out_idx] - self.timestamps[in_idx],
            'takeoff_angle_deg': takeoff_angle,
            'landing_angle_deg': landing_angle,
        }
    
    def _angle_to_ground(self, vec: np.ndarray) -> float:
        """Compute angle (degrees) of vector relative to ground plane (XZ)."""
        horizontal_component = np.linalg.norm(vec[[0, 2]])
        vertical_component = vec[1]  # Y is up
        angle_rad = np.arctan2(vertical_component, horizontal_component)
        return np.degrees(angle_rad)
```

---

## UI Requirements

### Core UI Elements

1. **Video scrubber** with draggable in/out markers
   - Shows current frame
   - Markers initialized to auto-detected trick start/end
   - User can adjust if prediction is off

2. **Trajectory preview** (side view or 3D)
   - Shows reconstructed path
   - Ground plane visible
   - In/out points marked with angle indicators
   - Updates live as user adjusts markers

3. **Board length input**
   - Default: 155 cm
   - User enters their actual board length for accurate scaling
   - Triggers metric recalculation on change

4. **Results display**
   - Horizontal distance (m)
   - Arc length / total distance (m)
   - Peak height (m)
   - Airtime (s)
   - Takeoff angle (°)
   - Landing angle (°)

### UI Mockup

```
┌───────────────────────────────────────────────────────────┐
│  Video Scrubber                                           │
│  [====●=======○===================●====]                  │
│       ↑ IN    ↑ current           ↑ OUT                   │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  Trajectory Preview                                       │
│                                                           │
│           * * *                                           │
│         *       *    ↗ 35°                                │
│        *          *                                       │
│   ────*────────────*────  (ground)                        │
│    25° ↗            ↘ -40°                                │
│       IN             OUT                                  │
└───────────────────────────────────────────────────────────┘

Board length: [155] cm

┌───────────────────────────────────────────────────────────┐
│  RESULTS                                                  │
│                                                           │
│  Horizontal distance:  12.4 m                             │
│  Arc length (3D):      14.1 m                             │
│  Peak height:           2.8 m                             │
│  Airtime:               1.2 s                             │
│  Takeoff angle:          25°                              │
│  Landing angle:         -40°                              │
└───────────────────────────────────────────────────────────┘
```

---

## Resolved Design Decisions

### 1. Technology Stack

**Backend (Python)**:
- MASt3R for 3D reconstruction
- SAM2 for video segmentation
- NumPy/SciPy for geometry (RANSAC, coordinate transforms)
- FastAPI or Flask for serving results to frontend

**Frontend (TypeScript)**:
- Vanilla TypeScript (no framework)
- HTML5 video element for playback
- Canvas or WebGL for trajectory visualization
- Simple, minimal UI

**Communication**:
- Upload video → backend processes → returns JSON results + trajectory data
- Frontend handles visualization and user adjustments (in/out points, board length)
- Recalculation of metrics can happen client-side (just math on trajectory data)

---

### 2. Board Length / Scale Calibration

**Decision**: User always provides board length manually.

- Default value: 155 cm
- UI: Simple input field
- No auto-detection (adds complexity, potential for error)
- User knows their board length; this is the most reliable source

---

### 3. Ground Plane Handling

**Decision**: Single RANSAC plane fit to snow/ground points, used for:

1. Defining "up" direction (Y-axis = plane normal)
2. Computing height above ground for feature detection
3. Calculating takeoff/landing angles

**Implementation**:
```python
def fit_ground_plane(pointmaps, rider_masks):
    """
    Fit plane to all non-rider points across all frames.
    Returns (a, b, c, d) for plane ax + by + cz + d = 0
    """
    ground_points = []
    for pointmap, mask in zip(pointmaps, rider_masks):
        # Points where rider is NOT (i.e., snow/ground)
        ground_points.append(pointmap[~mask])
    
    all_ground = np.vstack(ground_points)
    
    # RANSAC plane fitting
    plane = ransac_fit_plane(all_ground, threshold=0.1, iterations=1000)
    return plane
```

**Limitations accepted**:
- Assumes terrain is roughly planar (works for most park features)
- Step-downs and gap jumps with very different takeoff/landing heights may show slightly skewed angles
- For complex terrain, user can adjust in/out points to compensate

---

### 4. Coordinate System

**Decision**: Right-handed coordinate system derived from scene geometry.

- **Y-axis (up)**: Ground plane normal, pointing away from snow
- **Z-axis (forward)**: Direction of travel, computed from trajectory start→end, projected onto ground plane
- **X-axis (lateral)**: Cross product of Y and Z (right-hand rule)

**Implementation**:
```python
def compute_world_frame(ground_plane, trajectory_3d):
    """
    Compute world coordinate frame from ground plane and trajectory.
    Returns rotation matrix to transform points into world coords.
    """
    # Y = ground normal (ensure pointing "up" based on trajectory)
    normal = ground_plane[:3]
    y_axis = normal / np.linalg.norm(normal)
    
    # Check normal direction (should point toward rider, not into ground)
    rider_center = trajectory_3d.mean(axis=0)
    if np.dot(y_axis, rider_center) < 0:
        y_axis = -y_axis
    
    # Z = direction of travel projected onto ground
    travel_dir = trajectory_3d[-1] - trajectory_3d[0]
    travel_dir_ground = travel_dir - np.dot(travel_dir, y_axis) * y_axis
    z_axis = travel_dir_ground / np.linalg.norm(travel_dir_ground)
    
    # X = Y × Z (lateral)
    x_axis = np.cross(y_axis, z_axis)
    
    return np.stack([x_axis, y_axis, z_axis], axis=1)  # 3×3 rotation
```

**Why this works**:
- Horizontal distance is meaningful (distance in XZ plane)
- Height is meaningful (Y component)
- Angles are relative to actual ground, not camera orientation
- Works regardless of camera position/angle

---

## Open Questions / TBD

### 1. Deployment / Compute

**Problem**: MASt3R and SAM2 are GPU-hungry.

**Options**:
- A) Local processing (user needs GPU)
- B) Cloud processing (FLAME, Modal, Replicate, etc.)
- C) Hybrid (SAM2 in browser via WebGPU, MASt3R in cloud)

**Considerations**:
- Video upload latency for cloud
- Cost per analysis
- Target user (enthusiast vs. mass market)

**Decision**: TBD

---

## Implementation Order (Suggested)

### Phase 1: Core Pipeline (CLI proof of concept)
1. [ ] Set up MASt3R inference (input: frames, output: pointmaps)
2. [ ] Set up SAM2 video segmentation (input: video + initial point, output: masks)
3. [ ] Implement mask → pointmap → trajectory fusion
4. [ ] Implement ground plane RANSAC
5. [ ] Implement coordinate frame computation (Y-up, Z-forward)
6. [ ] Implement feature segment detection (height threshold)
7. [ ] Implement distance and angle calculations
8. [ ] Test on sample trick video (jump and rail)

### Phase 2: Scale Calibration
9. [ ] Implement board measurement in 3D (segment board, measure extent)
10. [ ] Implement scale correction with user-provided board length
11. [ ] Validate accuracy against known distances (if possible)

### Phase 3: UI (Vanilla TypeScript)
12. [ ] Set up project structure (HTML, TypeScript, build tooling)
13. [ ] Video upload and playback (HTML5 video element)
14. [ ] Scrubber with draggable in/out markers
15. [ ] Trajectory visualization (Canvas 2D side-view with ground plane and angles)
16. [ ] Board length input (default 155cm)
17. [ ] Results display (all metrics)
18. [ ] Client-side metric recalculation on user adjustments

### Phase 4: Polish
19. [ ] Error handling (bad video, lost tracking, etc.)
20. [ ] Progress indication (these models are slow)
21. [ ] Export/share results

---

## References

- **MASt3R**: [GitHub](https://github.com/naver/mast3r) | [Paper](https://arxiv.org/abs/2406.09756)
- **DUSt3R**: [GitHub](https://github.com/naver/dust3r) | [Paper](https://arxiv.org/abs/2312.14132)
- **SAM2**: [GitHub](https://github.com/facebookresearch/segment-anything-2) | [Paper](https://arxiv.org/abs/2408.00714)

---

## Notes

- Target accuracy: ~90% (acceptable for recreational/coaching use)
- Primary use case: Snowboard trick analysis from phone/action cam footage
- Supported features: Jumps, rails, boxes, any terrain park feature
- Invariant to: Spins, flips, grabs, and other rider rotations
- User provides: Video file, board length (default 155cm)
- System provides: Distance metrics, height, airtime, angles, trajectory visualization
