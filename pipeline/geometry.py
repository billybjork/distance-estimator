"""
Geometry processing for snowboard trick analysis.

Handles:
- Mask-pointmap fusion (extracting rider 3D trajectory)
- Ground plane fitting (RANSAC)
- Coordinate frame computation (Y-up, Z-forward)
- Distance and angle calculations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TrickAnalysis:
    """
    Results of trick analysis from the spec.

    All coordinates are in world frame (Y-up, Z-forward).
    """
    trajectory_3d: np.ndarray      # N×3 positions in world coords
    heights: np.ndarray            # N floats (height above ground)
    timestamps: np.ndarray         # N floats (seconds from video start)
    frame_indices: np.ndarray      # N ints (original frame numbers)
    predicted_in: int              # auto-detected trick start (index into arrays)
    predicted_out: int             # auto-detected trick end (index into arrays)
    ground_plane: np.ndarray       # plane coefficients (a, b, c, d) for ax+by+cz+d=0
    board_length_3d: float         # measured board length in reconstruction units

    def compute_metrics(
        self,
        in_idx: Optional[int] = None,
        out_idx: Optional[int] = None,
        board_length_cm: float = 155.0,
    ) -> dict:
        """
        Compute trick metrics with optional user overrides.

        Args:
            in_idx: Override for trick start index
            out_idx: Override for trick end index
            board_length_cm: Known board length in cm for scale calibration

        Returns:
            Dictionary with all computed metrics
        """
        in_idx = in_idx if in_idx is not None else self.predicted_in
        out_idx = out_idx if out_idx is not None else self.predicted_out

        # Scale factor: known board length / measured board length
        scale = (board_length_cm / 100) / self.board_length_3d if self.board_length_3d > 0 else 1.0

        segment = self.trajectory_3d[in_idx:out_idx + 1] * scale
        heights = self.heights[in_idx:out_idx + 1] * scale

        if len(segment) < 2:
            return {
                'horizontal_m': 0.0,
                'arc_length_m': 0.0,
                'peak_height_m': 0.0,
                'airtime_s': 0.0,
                'takeoff_angle_deg': 0.0,
                'landing_angle_deg': 0.0,
            }

        # Horizontal distance (start to end, XZ plane)
        horizontal = np.linalg.norm(segment[-1, [0, 2]] - segment[0, [0, 2]])

        # Arc length (sum of consecutive 3D distances)
        deltas = np.diff(segment, axis=0)
        arc_length = np.sum(np.linalg.norm(deltas, axis=1))

        # Takeoff angle (angle of initial velocity relative to ground)
        takeoff_vec = segment[1] - segment[0]
        takeoff_angle = self._angle_to_ground(takeoff_vec)

        # Landing angle (angle of final velocity relative to ground)
        landing_vec = segment[-1] - segment[-2]
        landing_angle = self._angle_to_ground(landing_vec)

        return {
            'horizontal_m': float(horizontal),
            'arc_length_m': float(arc_length),
            'peak_height_m': float(heights.max()),
            'airtime_s': float(self.timestamps[out_idx] - self.timestamps[in_idx]),
            'takeoff_angle_deg': float(takeoff_angle),
            'landing_angle_deg': float(landing_angle),
        }

    def _angle_to_ground(self, vec: np.ndarray) -> float:
        """Compute angle (degrees) of vector relative to ground plane (XZ)."""
        horizontal_component = np.linalg.norm(vec[[0, 2]])
        vertical_component = vec[1]  # Y is up
        angle_rad = np.arctan2(vertical_component, horizontal_component)
        return np.degrees(angle_rad)


def fuse_masks_and_pointmaps(
    masks: List[np.ndarray],
    pointmaps: List[np.ndarray],
    confidences: List[np.ndarray],
    min_confidence: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse SAM2 masks with MASt3R pointmaps to extract rider 3D positions.

    Args:
        masks: List of H×W boolean/uint8 masks from SAM2
        pointmaps: List of H×W×3 pointmaps from MASt3R (may be flattened)
        confidences: List of H×W confidence maps from MASt3R
        min_confidence: Minimum confidence threshold for points

    Returns:
        Tuple of:
            - rider_positions: N×3 array of rider centroid positions
            - all_ground_points: M×3 array of all non-rider points (for plane fitting)
    """
    rider_positions = []
    all_ground_points = []

    for i, (mask, pointmap, conf) in enumerate(zip(masks, pointmaps, confidences)):
        # Handle flattened pointmaps (N×3 instead of H×W×3)
        if pointmap.ndim == 2 and pointmap.shape[1] == 3:
            # Pointmap is flattened, need to reshape
            # Assume square for MASt3R output (typically 384x384 or 512x512)
            n_points = pointmap.shape[0]
            side = int(np.sqrt(n_points))
            if side * side == n_points:
                pointmap = pointmap.reshape(side, side, 3)
            else:
                # Non-square, try to match mask dimensions
                h, w = mask.shape[:2]
                if h * w == n_points:
                    pointmap = pointmap.reshape(h, w, 3)

        # Handle flattened confidence maps
        if conf is not None and conf.ndim == 1:
            n_conf = conf.shape[0]
            side = int(np.sqrt(n_conf))
            if side * side == n_conf:
                conf = conf.reshape(side, side)

        # Ensure mask matches pointmap dimensions
        pm_h, pm_w = pointmap.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (mask_h, mask_w) != (pm_h, pm_w):
            # Resize mask to match pointmap
            from scipy.ndimage import zoom
            scale_h = pm_h / mask_h
            scale_w = pm_w / mask_w
            mask = zoom(mask.astype(float), (scale_h, scale_w), order=0) > 0.5

        # Ensure confidence matches pointmap dimensions
        if conf is not None:
            conf_h, conf_w = conf.shape[:2]
            if (conf_h, conf_w) != (pm_h, pm_w):
                from scipy.ndimage import zoom
                scale_h = pm_h / conf_h
                scale_w = pm_w / conf_w
                conf = zoom(conf.astype(float), (scale_h, scale_w), order=1)

        # Create boolean mask
        mask_bool = mask.astype(bool)

        # Apply confidence threshold
        conf_mask = conf > min_confidence if conf is not None else np.ones_like(mask_bool)

        # Extract rider points (masked AND confident)
        rider_mask = mask_bool & conf_mask
        if rider_mask.sum() > 0:
            rider_points = pointmap[rider_mask]
            # Compute centroid
            centroid = rider_points.mean(axis=0)
            rider_positions.append(centroid)
        else:
            # Fallback: use mask center even without confidence
            if mask_bool.sum() > 0:
                rider_points = pointmap[mask_bool]
                centroid = rider_points.mean(axis=0)
                rider_positions.append(centroid)
            else:
                # No mask at all, skip this frame
                rider_positions.append(np.array([np.nan, np.nan, np.nan]))

        # Extract ground points (not masked AND confident)
        ground_mask = (~mask_bool) & conf_mask
        if ground_mask.sum() > 0:
            ground_points = pointmap[ground_mask]
            # Subsample to avoid memory issues
            if len(ground_points) > 10000:
                indices = np.random.choice(len(ground_points), 10000, replace=False)
                ground_points = ground_points[indices]
            all_ground_points.append(ground_points)

    rider_positions = np.array(rider_positions)

    if all_ground_points:
        all_ground_points = np.vstack(all_ground_points)
    else:
        all_ground_points = np.zeros((0, 3))

    return rider_positions, all_ground_points


def fit_ground_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.1,
    iterations: int = 1000,
    min_inliers: int = 100,
) -> np.ndarray:
    """
    Fit a plane to points using RANSAC.

    Args:
        points: N×3 array of 3D points
        threshold: Distance threshold for inliers
        iterations: Number of RANSAC iterations
        min_inliers: Minimum inliers for valid plane

    Returns:
        Plane coefficients (a, b, c, d) where ax + by + cz + d = 0
        Normal is (a, b, c) with unit length
    """
    if len(points) < 3:
        # Default: horizontal plane at y=0
        return np.array([0.0, 1.0, 0.0, 0.0])

    best_plane = None
    best_inliers = 0

    for _ in range(iterations):
        # Sample 3 random points
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-10:
            continue

        normal = normal / norm_length
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d])

    if best_plane is None or best_inliers < min_inliers:
        # Default: horizontal plane at median y
        median_y = np.median(points[:, 1])
        return np.array([0.0, 1.0, 0.0, -median_y])

    return best_plane


def compute_world_frame(
    ground_plane: np.ndarray,
    trajectory_3d: np.ndarray,
) -> np.ndarray:
    """
    Compute world coordinate frame from ground plane and trajectory.

    Args:
        ground_plane: Plane coefficients (a, b, c, d)
        trajectory_3d: N×3 trajectory positions

    Returns:
        3×3 rotation matrix to transform points into world coords
        Columns are [X, Y, Z] axes where Y is up and Z is forward
    """
    # Y = ground normal (ensure pointing "up" based on trajectory)
    normal = ground_plane[:3]
    y_axis = normal / np.linalg.norm(normal)

    # Check normal direction (should point toward rider, not into ground)
    # Use trajectory centroid to determine "up"
    rider_center = np.nanmean(trajectory_3d, axis=0)

    # Find a point on the plane
    if np.abs(y_axis[1]) > 0.5:  # Y component is significant
        plane_point = np.array([0, -ground_plane[3] / y_axis[1], 0])
    elif np.abs(y_axis[0]) > 0.5:
        plane_point = np.array([-ground_plane[3] / y_axis[0], 0, 0])
    else:
        plane_point = np.array([0, 0, -ground_plane[3] / y_axis[2]])

    # Vector from plane to rider should align with normal
    to_rider = rider_center - plane_point
    if np.dot(y_axis, to_rider) < 0:
        y_axis = -y_axis

    # Z = direction of travel projected onto ground
    valid_positions = trajectory_3d[~np.isnan(trajectory_3d).any(axis=1)]
    if len(valid_positions) < 2:
        # Default forward direction
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        travel_dir = valid_positions[-1] - valid_positions[0]
        # Project onto ground plane
        travel_dir_ground = travel_dir - np.dot(travel_dir, y_axis) * y_axis
        z_norm = np.linalg.norm(travel_dir_ground)
        if z_norm > 1e-10:
            z_axis = travel_dir_ground / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

    # X = Y × Z (lateral, right-hand rule)
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-10:
        x_axis = x_axis / x_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0])

    # Ensure orthonormal (recompute Z)
    z_axis = np.cross(x_axis, y_axis)

    return np.stack([x_axis, y_axis, z_axis], axis=1)  # 3×3 rotation


def transform_to_world(
    points: np.ndarray,
    rotation: np.ndarray,
    ground_plane: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform points to world coordinates and compute heights.

    Args:
        points: N×3 points in reconstruction coordinates
        rotation: 3×3 rotation matrix from compute_world_frame
        ground_plane: Plane coefficients (a, b, c, d)

    Returns:
        Tuple of:
            - world_points: N×3 points in world coordinates
            - heights: N array of heights above ground
    """
    # Handle NaN values
    valid_mask = ~np.isnan(points).any(axis=1)

    world_points = np.full_like(points, np.nan)
    heights = np.full(len(points), np.nan)

    if valid_mask.sum() == 0:
        return world_points, heights

    valid_points = points[valid_mask]

    # Compute heights (distance from ground plane)
    normal = ground_plane[:3]
    d = ground_plane[3]
    valid_heights = np.abs(np.dot(valid_points, normal) + d)

    # Transform to world coordinates
    # First, translate so plane passes through origin
    # Then rotate so Y is up
    valid_world = valid_points @ rotation

    world_points[valid_mask] = valid_world
    heights[valid_mask] = valid_heights

    return world_points, heights


def detect_feature_segment(
    heights: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[int, int]:
    """
    Find longest contiguous segment above height threshold.
    Works for both jumps (airborne) and rails (elevated).

    Args:
        heights: N array of heights above ground
        threshold: Height threshold in reconstruction units

    Returns:
        Tuple of (start_idx, end_idx) for the trick segment
    """
    # Handle NaN values
    valid_heights = np.nan_to_num(heights, nan=0.0)

    on_feature = valid_heights > threshold

    # Find transitions
    padded = np.concatenate([[False], on_feature, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        # No segments above threshold, use full range
        return 0, len(heights) - 1

    # Find longest segment
    lengths = ends - starts
    best = np.argmax(lengths)

    return int(starts[best]), int(ends[best] - 1)


def estimate_board_length(
    masks: List[np.ndarray],
    pointmaps: List[np.ndarray],
    confidences: List[np.ndarray],
) -> float:
    """
    Estimate snowboard length from masked pointmaps.

    Uses PCA to find the primary axis of the rider mask and measures extent.
    This is a rough estimate - the board is the longest rigid part of the rider.

    Args:
        masks: List of rider masks
        pointmaps: List of pointmaps
        confidences: List of confidence maps

    Returns:
        Estimated board length in reconstruction units
    """
    all_extents = []

    for mask, pointmap, conf in zip(masks, pointmaps, confidences):
        # Reshape if needed
        if pointmap.ndim == 2 and pointmap.shape[1] == 3:
            n_points = pointmap.shape[0]
            side = int(np.sqrt(n_points))
            if side * side == n_points:
                pointmap = pointmap.reshape(side, side, 3)
                conf = conf.reshape(side, side) if conf.ndim == 1 else conf

        # Resize mask if needed
        pm_h, pm_w = pointmap.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (mask_h, mask_w) != (pm_h, pm_w):
            from scipy.ndimage import zoom
            scale_h = pm_h / mask_h
            scale_w = pm_w / mask_w
            mask = zoom(mask.astype(float), (scale_h, scale_w), order=0) > 0.5

        # Get rider points
        mask_bool = mask.astype(bool)
        if mask_bool.sum() < 10:
            continue

        rider_points = pointmap[mask_bool]

        # PCA to find primary axis
        centered = rider_points - rider_points.mean(axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Primary axis is eigenvector with largest eigenvalue
            primary_axis = eigenvectors[:, -1]

            # Project points onto primary axis
            projections = np.dot(centered, primary_axis)
            extent = projections.max() - projections.min()
            all_extents.append(extent)
        except:
            continue

    if not all_extents:
        return 1.55  # Default ~155cm in meters

    # Use median extent as board length estimate
    return float(np.median(all_extents))


def analyze_trick(
    masks: List[np.ndarray],
    pointmaps: List[np.ndarray],
    confidences: List[np.ndarray],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    height_threshold: float = 0.3,
    min_confidence: float = 1.5,
) -> TrickAnalysis:
    """
    Full trick analysis pipeline.

    Args:
        masks: List of SAM2 rider masks
        pointmaps: List of MASt3R pointmaps
        confidences: List of MASt3R confidence maps
        timestamps: Array of frame timestamps
        frame_indices: Array of original frame indices
        height_threshold: Threshold for feature detection
        min_confidence: Minimum confidence for points

    Returns:
        TrickAnalysis object with all results
    """
    # Step 1: Fuse masks and pointmaps
    rider_positions, ground_points = fuse_masks_and_pointmaps(
        masks, pointmaps, confidences, min_confidence
    )

    # Step 2: Fit ground plane
    ground_plane = fit_ground_plane_ransac(ground_points)

    # Step 3: Compute world frame
    rotation = compute_world_frame(ground_plane, rider_positions)

    # Step 4: Transform to world coordinates
    world_trajectory, heights = transform_to_world(
        rider_positions, rotation, ground_plane
    )

    # Step 5: Detect trick segment
    predicted_in, predicted_out = detect_feature_segment(heights, height_threshold)

    # Step 6: Estimate board length
    board_length = estimate_board_length(masks, pointmaps, confidences)

    return TrickAnalysis(
        trajectory_3d=world_trajectory,
        heights=heights,
        timestamps=timestamps,
        frame_indices=frame_indices,
        predicted_in=predicted_in,
        predicted_out=predicted_out,
        ground_plane=ground_plane,
        board_length_3d=board_length,
    )
