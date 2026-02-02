import numpy as np
import yaml


class DepthEstimator:
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize depth estimator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.distance_config = self.config['distance']
        
        # Parameters
        self.min_range = self.distance_config['min_range']
        self.max_range = self.distance_config['max_range']
        self.max_depth_std = self.distance_config['max_depth_std']
        
        print(f"✓ Depth estimator initialized")
        print(f"  Range: {self.min_range}m - {self.max_range}m")
    
    def estimate_distance(self, detection, depth_image):
        """
        Estimate distance to detected object with improved robustness
        
        Args:
            detection: Detection dict with 'bbox' key
            depth_image: Depth image (H x W) in meters
        
        Returns:
            dict with:
                - distance: float (meters)
                - valid: bool
                - method: str
                - std_dev: float
                - reason: str (if invalid)
        """
        bbox = detection['bbox']
        x, y, w, h = bbox
        
        # Validate bbox size first
        if w < 20 or h < 20:
            return {
                'distance': None,
                'valid': False,
                'method': 'depth_buffer',
                'std_dev': None,
                'reason': 'bbox_too_small'
            }
        
        # Define ROI - use center region but ensure minimum size
        roi_fraction = 0.3  # Use center 50% of bbox
        min_roi_size = 15  # Minimum ROI dimension in pixels
        
        # Calculate ROI size
        w_roi = max(int(w * roi_fraction), min_roi_size)
        h_roi = max(int(h * roi_fraction), min_roi_size)
        
        # Center the ROI
        x_start = int(x + (w - w_roi) / 2)
        y_start = int(y + (h - h_roi) / 2)
        x_end = x_start + w_roi
        y_end = y_start + h_roi
        
        # Clip to image bounds
        height, width = depth_image.shape
        x_start = max(0, min(x_start, width - 1))
        x_end = max(x_start + 1, min(x_end, width))
        y_start = max(0, min(y_start, height - 1))
        y_end = max(y_start + 1, min(y_end, height))
        
        # Extract depth ROI
        depth_roi = depth_image[y_start:y_end, x_start:x_end]
        
        if depth_roi.size == 0:
            return {
                'distance': None,
                'valid': False,
                'method': 'depth_buffer',
                'std_dev': None,
                'reason': 'empty_roi'
            }
        
        # Filter outliers (keep only depths in valid range)
        valid_depths = depth_roi[
            (depth_roi >= self.min_range) & 
            (depth_roi <= self.max_range) &
            (np.isfinite(depth_roi))  # Remove inf/nan
        ]
        
        if len(valid_depths) == 0:
            return {
                'distance': None,
                'valid': False,
                'method': 'depth_buffer',
                'std_dev': None,
                'reason': 'no_valid_depths'
            }
        
        # Calculate statistics using median (robust to outliers)
        median_depth = np.median(valid_depths)
        std_depth = np.std(valid_depths)
        
        # Adaptive validation: std should be < 15% of distance
        # This is more lenient for far objects, stricter for near ones
        max_allowed_std = max(median_depth * 0.2, 0.08)
        # Adaptive validation: std should be < 15% of distance
        # This is more lenient for far objects, stricter for near ones
        max_allowed_std = max(median_depth * 0.12, 0.05)  # At least 5cm tolerance
        valid = std_depth < max_allowed_std

        # Additional validation: need enough valid pixels for confidence
        min_valid_pixels = 10
        if len(valid_depths) < min_valid_pixels:
            valid = False
        
        return {
            'distance': float(median_depth),
            'valid': valid,
            'method': 'depth_buffer',
            'std_dev': float(std_depth),
            'roi_size': (w_roi, h_roi),
            'num_valid_pixels': len(valid_depths)
        }
    
    def estimate_distance_geometric(self, detection, real_size, focal_length):
        """
        Fallback: Estimate distance using geometry (if depth not available)
        
        Args:
            detection: Detection dict
            real_size: Real-world size of object (meters)
            focal_length: Camera focal length (pixels)
        
        Returns:
            Distance in meters
        """
        bbox = detection['bbox']
        bbox_width = bbox[2]
        
        # distance = (real_size × focal_length) / pixel_size
        distance = (real_size * focal_length) / bbox_width
        
        return distance


class PositionCalculator:
    """Calculates 3D position from detection and distance"""
    
    def __init__(self, camera_intrinsics, config_path="config/config.yaml"):
        """
        Initialize position calculator
        
        Args:
            camera_intrinsics: Dict with fx, fy, cx, cy
            config_path: Path to config
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.position_config = self.config['position']
        
        # Camera parameters
        self.fx = camera_intrinsics.get('focal_length_x', camera_intrinsics.get('fx'))
        self.fy = camera_intrinsics.get('focal_length_y', camera_intrinsics.get('fy'))
        self.cx = camera_intrinsics.get('principal_point_x', camera_intrinsics.get('cx'))
        self.cy = camera_intrinsics.get('principal_point_y', camera_intrinsics.get('cy'))
        
        # Load camera pose
        cam_config = self.config['camera']
        self.cam_pos = np.array(cam_config['position'])
        self.cam_target = np.array(cam_config['target'])
        self.cam_up = np.array(cam_config['up'])
        
        # Compute camera-to-world rotation matrix
        self.R_cam_to_world = self._compute_rotation_matrix()
        
        print(f"✓ Position calculator initialized")
        print(f"  Focal length: ({self.fx:.1f}, {self.fy:.1f})")
        print(f"  Camera position: {self.cam_pos}")
    
    def _compute_rotation_matrix(self):
        """
        Compute rotation matrix from camera frame to world frame
        
        Camera frame (OpenGL/PyBullet convention):
            X: right
            Y: up  
            Z: backward (into the camera, negative of view direction)
        
        World frame:
            X, Y, Z as defined in config
        
        Returns:
            3x3 rotation matrix
        """
        # Forward vector (camera looking direction)
        forward = self.cam_target - self.cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Right vector
        right = np.cross(forward, self.cam_up)
        right = right / np.linalg.norm(right)
        
        # Up vector (recompute for orthogonality)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Rotation matrix: columns are camera axes expressed in world frame
        # Camera X-axis (right) -> right vector in world
        # Camera Y-axis (up) -> up vector in world  
        # Camera Z-axis (back) -> -forward vector in world
        R = np.column_stack([right, up, -forward])
        
        print(f"  Camera rotation matrix computed:")
        print(f"    Right: {right}")
        print(f"    Up: {up}")
        print(f"    Back: {-forward}")
        
        return R
    
    def calculate_3d_position(self, detection, distance):
        """
        Calculate 3D position using proper coordinate transformation
        
        Args:
            detection: Detection dict with 'center' key [cx_pixel, cy_pixel]
            distance: Distance in meters (depth from camera)
        
        Returns:
            dict with:
                - position: [x, y, z] in world frame (meters)
                - position_camera: [x, y, z] in camera frame
                - confidence: float
        """
        cx_pixel, cy_pixel = detection['center']
        
        # Convert pixel coordinates to normalized image coordinates
        x_norm = (cx_pixel - self.cx) / self.fx
        y_norm = (cy_pixel - self.cy) / self.fy
        
        # 3D point in camera frame
        # In camera frame (OpenGL convention used by PyBullet):
        #   X: right (positive = right of image center)
        #   Y: up (positive = up from image center, NOTE: image Y is down!)
        #   Z: backward (positive = into camera, so distance is negative Z)
        
        ray_dir_camera = np.array([
            x_norm,      # Right offset
            -y_norm,     # Up offset (negate because image Y is down)
            -1.0         # Forward direction (-Z in camera space)
        ])
        
        # Normalize to unit vector
        ray_dir_camera = ray_dir_camera / np.linalg.norm(ray_dir_camera)
        
        # Scale by distance to get 3D point in camera frame
        # This gives us the point at 'distance' meters along the ray
        p_camera = ray_dir_camera * distance
        
        # Transform to world frame
        # p_world = camera_position + R * p_camera
        p_world = self.cam_pos + self.R_cam_to_world @ p_camera
        
        # Estimate confidence based on distance and detection confidence
        confidence = self._estimate_confidence(detection, distance)
        
        return {
            'position': p_world.tolist(),
            'position_camera': p_camera.tolist(),
            'confidence': float(confidence)
        }
    
    def _estimate_confidence(self, detection, distance):
        """
        Estimate position confidence based on detection confidence and distance
        
        Args:
            detection: Detection dict with 'confidence'
            distance: Distance in meters
        
        Returns:
            Confidence score (0-1)
        """
        det_conf = detection['confidence']
        
        # Confidence decreases with distance (harder to localize far objects)
        if distance < 1.0:
            dist_factor = 1.0
        elif distance < 2.0:
            dist_factor = 0.9
        elif distance < 3.0:
            dist_factor = 0.7
        else:
            dist_factor = 0.5
        
        return float(det_conf * dist_factor)
    
    def project_to_ground(self, position_3d):
        """
        Project 3D position onto ground plane
        
        Args:
            position_3d: [x, y, z] in world frame
        
        Returns:
            [x, y, z_ground] - position on ground
        """
        ground_height = self.position_config['ground_plane_height']
        return [position_3d[0], position_3d[1], ground_height]
    
    