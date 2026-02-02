# 🤖 Robot Vision System - Object Detection & 3D Localization

A complete computer vision system for detecting objects (cube, sphere, cylinder) in 3D space using PyBullet simulation, YOLOv8, and depth estimation. Perfect for robotics projects, pick-and-place tasks, and autonomous navigation.

## 🎯 What Does This Do?

This system:
- **Detects** objects in a simulated camera view using YOLOv8
- **Measures** distance to each object using depth buffers
- **Calculates** 3D position (X, Y, Z coordinates) in world space
- **Provides** accurate target coordinates for robot path planning

Perfect for robotics students and researchers working on object manipulation and navigation tasks!

---

## 📦 Project Structure

```
📦Detect
 ┣ 📂config
 ┃ ┗ 📜config.yaml              # All system parameters
 ┣ 📂scripts
 ┃ ┣ 📜test_detection.py        # Test detection accuracy
 ┃ ┗ 📜test_path.py             # Test path planning interface
 ┣ 📂src
 ┃ ┣ 📂detection
 ┃ ┃ ┗ 📜detector.py            # YOLO object detector
 ┃ ┣ 📂perception
 ┃ ┃ ┗ 📜depth_estimator.py     # Distance & position calculation
 ┃ ┣ 📂simulation
 ┃ ┃ ┣ 📜camera.py              # PyBullet camera interface
 ┃ ┃ ┗ 📜environment.py         # Simulation environment
 ┃ ┣ 📂tracking
 ┃ ┃ ┗ 📜object_tracker.py      # Object tracking (optional)
 ┃ ┗ 📜vision_system.py         # Main vision system
 ┣ 📂weights
 ┃ ┗ 📜best.pt                  # Trained YOLOv8 model (2000 images)
 ┗ 📜requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Detect

# Create conda virtual environment
conda create -n robot_vision python=3.11
conda activate robot_vision

# Install dependencies
pip install -r requirements.txt
```

**Note:** This project was developed using conda for environment management. Python 3.11 is recommended.

### 2. Test the System

**Test Detection Accuracy:**
```bash
python scripts/test_detection.py --scenes 5 --objects 3
```

**Test Path Planning Interface:**
```bash
python scripts/test_path.py
```

### 3. Use in Your Project

```python
import sys
sys.path.append('src')

from simulation.environment import RobotEnvironment
from simulation.camera import Camera
from vision_system import VisionSystem

# Setup
env = RobotEnvironment(gui=True)
env.spawn_random_scene(num_each_type=2)

camera = Camera()
vision = VisionSystem(camera, model_path="weights/best.pt")

# Run detection
result = vision.detect_and_measure()

# Get target coordinates for path planning
for detection in result['detections']:
    target_x = detection['position'][0]  # X coordinate
    target_y = detection['position'][1]  # Y coordinate
    object_type = detection['class_name']
    
    print(f"{object_type} at [{target_x:.3f}, {target_y:.3f}]")
```

---

## 🎓 How It Works

### 1. **Object Detection** (YOLOv8)
- Trained on 2000 synthetic images
- Detects: `cube`, `sphere`, `cylinder`
- Returns bounding boxes and confidence scores

### 2. **Distance Estimation** (Depth Buffer)
- Uses PyBullet's depth buffer
- Converts depth to real-world meters
- Validates measurements for accuracy

### 3. **3D Position Calculation**
- Transforms pixel coordinates to 3D world coordinates
- Uses camera intrinsics (focal length, principal point)
- Accounts for camera pose and orientation

### 4. **Coordinate System**
```
World Frame:
  X-axis: Left/Right
  Y-axis: Forward/Back  
  Z-axis: Up/Down

Camera Frame:
  Position: [0, -1.8, 0.6] (behind workspace, elevated)
  Looking at: [0, 0, 0.4] (table height)
```

---

## 📊 Performance Metrics

**Trained Model Performance:**
- Training Dataset: 2000 images
- Distance Range: 0.8m - 1.8m
- Angle Range: -15° to +15°

**Expected Accuracy:**
- F1 Score Target: ≥ 80%
- Distance Error: < 15cm
- Position Error: < 20cm

---

## ⚙️ Configuration

All parameters are in `config/config.yaml`:

### Object Definitions
```yaml
objects:
  sphere:
    diameter: 0.15
    color: [0.0, 0.0, 1.0, 1.0]  # Blue
    class_id: 1
  cube:
    size: [0.20, 0.20, 0.20]
    color: [1.0, 0.0, 0.0, 1.0]  # Red
    class_id: 0
  cylinder:
    height: 0.16
    radius: 0.06
    color: [0.0, 1.0, 0.0, 1.0]  # Green
    class_id: 2
```

### Camera Settings
```yaml
camera:
  width: 640
  height: 480
  fov: 60
  position: [0, -1.8, 0.6]
  target: [0, 0, 0.4]
```

### Detection Parameters
```yaml
detection:
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

---

## 🔧 Advanced Usage

### Customize Object Spawning
```python
# Spawn objects at specific distances and angles
env.spawn_object('cube', position=[1.0, 0.5, 0.4])
env.spawn_object('sphere', position=[1.5, -0.3, 0.3])
```

### Access Detailed Results
```python
result = vision.detect_and_measure()

for det in result['detections']:
    print(f"Class: {det['class_name']}")
    print(f"Confidence: {det['confidence']:.2%}")
    print(f"Distance: {det['distance']:.3f}m")
    print(f"Position: {det['position']}")  # [x, y, z]
    print(f"BBox: {det['bbox']}")  # [x, y, w, h] in pixels
```

### Visualize Detections
```python
import cv2

result = vision.detect_and_measure()
vis_image = vision.visualize(result)

cv2.imshow("Detections", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

---

## 📝 API Reference

### VisionSystem Class

**Main Methods:**

```python
# Initialize vision system
vision = VisionSystem(camera, model_path="weights/best.pt")

# Run detection and measurement
result = vision.detect_and_measure()
# Returns: {
#   'detections': [...],      # List of detected objects
#   'rgb_image': ndarray,     # Original RGB image
#   'depth_image': ndarray,   # Depth buffer
#   'num_detections': int,    # Number of objects found
#   'processing_time': float  # Time taken (seconds)
# }

# Create visualization
vis_image = vision.visualize(result)

# Get system performance
fps = vision.get_fps()
detector_fps = vision.get_detector_fps()
```

### Detection Dictionary Format

Each detection contains:
```python
{
    'class_id': 0,                    # 0=cube, 1=sphere, 2=cylinder
    'class_name': 'cube',             # Object type
    'confidence': 0.95,               # Detection confidence (0-1)
    'bbox': [x, y, w, h],            # Bounding box in pixels
    'center': [cx, cy],              # Center point in pixels
    'distance': 1.234,               # Distance from camera (meters)
    'distance_std': 0.023,           # Distance uncertainty
    'position': [x, y, z],           # 3D position in world frame
    'position_camera': [x, y, z],    # 3D position in camera frame
    'position_confidence': 0.85      # Position confidence (0-1)
}
```

---

## 🧪 Testing

### Run Full Test Suite
```bash
# Test with 10 scenes, 5 objects each
python scripts/test_detection.py --scenes 10 --objects 5

# Test without visualization (faster)
python scripts/test_detection.py --scenes 20 --objects 3 --no-viz
```

### Expected Test Output
```
TEST RESULTS
======================================================================
Overall Performance:
  Total objects: 50
  Detected: 45
  Missed: 5
  False positives: 2

Metrics:
  Precision: 95.74%
  Recall: 90.00%
  F1 Score: 92.78%

Distance Estimation:
  Mean error: 0.089m
  Std dev: 0.067m
  Max error: 0.234m
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'src'"**
```bash
# Make sure you're running from project root
cd Detect
python scripts/test_detection.py
```

**2. "Cannot find trained model"**
```bash
# Check if weights/best.pt exists
ls weights/best.pt

# If missing, train a new model or use pretrained YOLOv8
```

**3. Objects not detected**
- Check if objects are in camera view (0.8m - 1.8m range)
- Verify camera position in config.yaml
- Ensure lighting is adequate in simulation

**4. Poor distance accuracy**
- Objects too close (<0.7m) or too far (>2.0m)
- Adjust `spawning.distances` in config.yaml
- Check depth buffer visualization

**5. Conda environment issues**
```bash
# Make sure conda environment is activated
conda activate robot_vision

# Verify Python version
python --version  # Should be 3.11.x

# Reinstall dependencies if needed
pip install -r requirements.txt --force-reinstall
```

---

## 🎯 Use Cases

### 1. **Pick and Place Robot**
```python
# Detect objects
result = vision.detect_and_measure()

# Plan path to each object
for obj in result['detections']:
    target = [obj['position'][0], obj['position'][1]]
    robot.move_to(target)
    robot.pick_up(obj['class_name'])
```

### 2. **Object Sorting**
```python
# Detect and sort by type
cubes = [d for d in detections if d['class_name'] == 'cube']
spheres = [d for d in detections if d['class_name'] == 'sphere']

# Process each category
for cube in cubes:
    robot.move_to_bin('cube_bin', cube['position'])
```

### 3. **Navigation Planning**
```python
# Detect obstacles
obstacles = vision.detect_and_measure()['detections']

# Get obstacle positions for path planner
obstacle_positions = [d['position'][:2] for d in obstacles]
path = path_planner.plan(start, goal, obstacles=obstacle_positions)
```

---

## 📚 Dependencies

**Environment Management:**
- **Conda**: This project uses conda for virtual environment management
- **Python**: 3.11 (recommended)

Main libraries:
- **PyBullet** (≥3.2.5): Physics simulation
- **YOLOv8/Ultralytics** (≥8.0.0): Object detection
- **OpenCV** (≥4.8.0): Image processing
- **NumPy** (≥1.24.0): Numerical computing
- **PyTorch** (≥2.0.0): Deep learning backend
- **PyYAML** (≥6.0): Configuration files

See `requirements.txt` for complete list.

**Managing your environment:**
```bash
# Activate environment
conda activate robot_vision

# Deactivate when done
conda deactivate

# Remove environment (if needed)
conda env remove -n robot_vision
```

---

## 🤝 Contributing

This is a personal project, but feel free to:
- Report bugs or issues
- Suggest improvements
- Fork and modify for your needs
- Share your results!

---

## 📄 License

[Add your license here - MIT, Apache 2.0, etc.]

---

## 👨‍💻 Author

Created for robotics research and education. Trained on 2000 synthetic images generated in PyBullet simulation.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **PyBullet** for physics simulation
- **PyTorch** for deep learning framework

---

## 📧 Contact

For questions or collaboration:
- GitHub: [Your GitHub]
- Email: [Your Email]

---

**Happy Robot Vision!** 🤖👁️✨s