# Fire Extinguisher 3D Reconstruction & Volume Estimation

## Overview
This project implements a complete pipeline for 3D reconstruction and volume estimation of fire extinguishers from RGB image sequences. The pipeline combines photogrammetry (COLMAP), point cloud processing (Open3D), and color-based segmentation to automatically detect and measure fire extinguisher volumes.

**Pipeline Steps:**
1. **Reconstruction** - COLMAP-based Structure-from-Motion (SfM) to generate 3D point clouds
2. **Denoising** - Statistical and radius-based outlier removal (3 methods available)
3. **Segmentation** - HSV color filtering and DBSCAN clustering to isolate the red extinguisher
4. **Volume Estimation** - Convex hull and adjusted cylinder volume calculation

**Target Accuracy:** 60-65L ±30% (42-84.5L acceptable range)

---

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **GPU**: Optional (CUDA-enabled GPU speeds up SIFT feature extraction)

### Dependencies
All required Python packages are listed in `requirements.txt`:
- `numpy` - Numerical computing
- `scipy` - Scientific computing utilities
- `open3d` - 3D point cloud processing and visualization
- `pycolmap` - COLMAP bindings for Structure-from-Motion
- `matplotlib` - Optional visualization utilities

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CV_project
```

### 2. Set Up Virtual Environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Input Data
Place your RGB images in the `raw/test/camera_color_image_raw/` directory. The pipeline expects:
- Sequential RGB images (`.png` format recommended)
- Camera calibration info in `raw/test/camera_color_camera_info/` (optional - default intrinsics used if missing)

---

## Usage

### Quick Start - Full Pipeline
Run the complete pipeline (reconstruction → denoising → segmentation → volume):
```bash
python main.py
```

This will:
1. Generate 3D point cloud from images (`sparse/scene/extinguisher_raw.ply`)
2. Denoise the point cloud (`sparse/scene/extinguisher_denoised_final.ply`)
3. Segment the red extinguisher (`extinguisher/extinguisher_clean.ply`)
4. Estimate volume with visualization

### Run Individual Modules
Execute specific pipeline steps independently:

**Reconstruction only:**
```bash
python main.py --module reconstruction
```

**Denoising only:**
```bash
python main.py --module denoising
```

**Segmentation only:**
```bash
python main.py --module segmentation
```

**Volume estimation only:**
```bash
python main.py --module volume
```

### Configuration
Edit `config.py` to adjust pipeline parameters:

**Camera Intrinsics:**
```python
CAMERA_FX = 306.0      # Focal length X
CAMERA_FY = 306.1      # Focal length Y
CAMERA_CX = 318.5      # Principal point X
CAMERA_CY = 201.4      # Principal point Y
```

**Denoising Methods:**
```python
DENOISING_METHOD = 3   # 1=SOR, 2=ROR, 3=COMBO
```

**HSV Color Filtering (Red Detection):**
```python
RED_H_LOW_MAX = 0.05   # Hue upper bound (red low)
RED_H_HIGH_MIN = 0.96  # Hue lower bound (red high)
RED_S_MIN = 0.10       # Saturation minimum
RED_V_MIN = 0.10       # Value minimum
```

**Volume Estimation:**
```python
CYLINDER_RADIUS_FACTOR = 0.7         # Radius adjustment
CYLINDER_HEIGHT_MARGIN_TOP = 0.05    # Top margin (meters)
CYLINDER_HEIGHT_MARGIN_BOTTOM = 0.05 # Bottom margin (meters)
```

**Visualization:**
```python
VISUALIZE = True       # Enable/disable Open3D visualizations
```

---

## Project Structure

```
CV_project/
├── config.py                    # Centralized configuration
├── main.py                      # Pipeline entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── modules/                     # Pipeline modules
│   ├── __init__.py
│   ├── reconstruction.py        # Module 1: COLMAP SfM
│   ├── denoising.py            # Module 2: Point cloud denoising
│   ├── segmentation.py         # Module 3: Red object segmentation
│   └── volume.py               # Module 4: Volume estimation
│
├── raw/                         # Input data
│   └── test/
│       ├── camera_color_image_raw/      # RGB images
│       └── camera_color_camera_info/    # Camera calibration
│
├── colmap/                      # COLMAP database (generated)
├── sparse/                      # Reconstruction outputs
│   ├── 0/                       # COLMAP sparse reconstruction
│   └── scene/                   # Point cloud outputs
│       ├── extinguisher_raw.ply
│       ├── extinguisher_denoised.ply
│       └── extinguisher_denoised_final.ply
│
├── scene/                       # Segmentation outputs
│   └── extinguisher_segment.ply
│
└── extinguisher/                # Final outputs
    └── extinguisher_clean.ply   # Cleaned segmented cloud
```

---

## Output Files

| File | Description | Step |
|------|-------------|------|
| `sparse/0/` | COLMAP sparse reconstruction (cameras, points) | 1 |
| `sparse/scene/extinguisher_raw.ply` | Raw reconstructed point cloud | 1 |
| `sparse/scene/extinguisher_denoised_final.ply` | Denoised point cloud | 2 |
| `scene/extinguisher_segment.ply` | Raw segmented extinguisher | 3 |
| `extinguisher/extinguisher_clean.ply` | Cleaned segmented extinguisher | 3 |
| Terminal output | Volume estimation (convex hull & cylinder) | 4 |

---

## Troubleshooting

**Issue: "Failed to load point cloud"**
- Check that input files exist at the paths specified in `config.py`
- Verify file permissions and directory structure

**Issue: "No red points detected"**
- Adjust HSV thresholds in `config.py` (RED_H_*, RED_S_MIN, RED_V_MIN)
- Ensure lighting conditions are adequate for red color detection

**Issue: "COLMAP reconstruction failed"**
- Verify image quality (not blurry, sufficient overlap between frames)
- Check camera intrinsics match your actual camera
- Try switching between SEQUENTIAL and EXHAUSTIVE matching methods

**Issue: Poor volume estimation**
- Adjust `CYLINDER_RADIUS_FACTOR` in `config.py`
- Tune segmentation parameters (DBSCAN eps, min_points)
- Try different denoising methods (1-3)

---

## Dataset Information

### Input Data Structure
The `raw/test/` directory contains data extracted from ROS1 rosbags captured with an Orbec camera:

**Camera Info Files:**
- `camera_color_camera_info/`, `camera_depth_camera_info/` - Calibration data (`.txt`)

**Images:**
- `camera_color_image_raw/` - RGB images (`.png`)
- `camera_depth_image_raw/` - Depth images (`.png`)
- `camera_left_ir_image_raw/` - Left infrared (`.png`)
- `camera_right_ir_image_raw/` - Right infrared (`.png`)

**Point Clouds:**
- `depth_registered_colored_pointclouds/` - Colored point clouds (`.pcd`)

### Data Format
- **PNG** - Lossless image format for RGB, depth, and IR images
- **PCD** - Point Cloud Data format (Open3D compatible)
- **PLY** - Polygon File Format for 3D meshes and point clouds

---

## Applications

This pipeline can be adapted for:
- **Quality Control** - Automated volume verification in manufacturing
- **Inventory Management** - 3D scanning and measurement of cylindrical objects
- **Robotics** - Object detection and volumetric analysis for manipulation
- **Computer Vision Research** - Benchmarking segmentation and reconstruction algorithms

---

## License

[Specify your license here]

## Authors

[Your name/team]

## Acknowledgments

- COLMAP for Structure-from-Motion reconstruction
- Open3D for point cloud processing
- Orbec camera for data capture