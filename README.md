# Fire Extinguisher 3D Reconstruction & Volume Estimation

## Overview
This project implements a complete pipeline for 3D reconstruction and volume estimation of fire extinguishers from RGB image sequences. The pipeline combines photogrammetry (COLMAP), point cloud processing (Open3D), and color-based segmentation to automatically detect and measure fire extinguisher volumes.

**Pipeline Steps:**
1. **Reconstruction** - COLMAP-based Structure-from-Motion (SfM) to generate 3D point clouds
2. **Denoising** - Statistical and radius-based outlier removal (3 methods available)
3. **Segmentation** - HSV color filtering and DBSCAN clustering to isolate the red extinguisher
4. **Volume Estimation** - Convex hull, PCA-aligned cylinder, and average volume calculation

**Volume Methods:**
- **Convex Hull** - Minimum bounding volume (tends to underestimate)
- **Cylinder** - PCA-aligned cylindrical approximation (tends to overestimate)
- **Average** - Mean of convex hull and cylinder volumes (recommended for final estimate)

**Target Accuracy:** 42.0 - 84.5 L acceptable range

---

## Requirements

### System Requirements
- **Python**: 3.9, 3.10, 3.11, or 3.12 (NOT 3.13 - open3d not yet supported)
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **GPU**: Optional (CUDA-enabled GPU speeds up SIFT feature extraction)

### Dependencies
All required Python packages are listed in `requirements.txt`:
- `numpy` - Numerical computing
- `scipy` - Scientific computing utilities
- `open3d` - 3D point cloud processing and visualization
- `pycolmap` - COLMAP bindings for Structure-from-Motion
- `matplotlib` - Visualization utilities
- `seaborn` - Statistical data visualization
- `pandas` - Data analysis (for benchmark pipeline)

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CV_project
```

### 2. Check Python Version
```bash
python --version
# Should be 3.9.x, 3.10.x, 3.11.x, or 3.12.x
# If you have Python 3.13, install Python 3.12 instead
```

### 3. Set Up Virtual Environment

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

**Note:** Before running the benchmark pipeline, set `VISUALIZE = False` in `config.py` to disable interactive visualizations.

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
4. Estimate volume (convex hull, cylinder, and average) with visualization

**Expected Output:**
```
VOLUME ESTIMATION RESULTS:
   • Convex Hull:     45.5 L
   • Cylinder:        67.5 L
   • Average:         56.5 L

Target Range: 42.0 - 84.5 L
   Status: WITHIN TARGET
```

### Benchmark Pipeline
Run multiple iterations to analyze reconstruction variability:

```bash
python benchmark_pipeline.py
```

**Options:**
- `-n`, `--iterations` - Number of iterations (default: 10)
- `-o`, `--output` - Output directory (default: benchmark_results)
- `--no-plots` - Skip generating plots

**Examples:**
```bash
# Run 20 iterations
python benchmark_pipeline.py -n 20

# Custom output directory
python benchmark_pipeline.py -n 15 -o results/benchmark_2025

# Skip plots (faster)
python benchmark_pipeline.py -n 5 --no-plots
```

**Output:**
- `benchmark_results.csv` - Raw data for all iterations
- `benchmark_summary.txt` - Statistical summary
- `volume_distribution.png` - Box plots of volume distributions
- `volume_time_series.png` - Volume measurements across iterations
- `volume_histogram.png` - Histograms with mean values
- `volume_correlation.png` - Convex hull vs cylinder scatter plot

### Point Cloud Viewer
Quickly visualize any point cloud file:

```bash
python main.py --view path/to/file.ply
```

**Example:**
```bash
python main.py -v sparse/scene/extinguisher_raw.ply
python main.py -v extinguisher/extinguisher_clean.ply
```

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
├── benchmark_pipeline.py        # Benchmark analysis tool
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
│   ├── scene/                   # Point cloud outputs
│   │   ├── extinguisher_raw.ply
│   │   ├── extinguisher_denoised.ply
│   │   └── extinguisher_denoised_final.ply
│   └── extinguisher/            # Final segmented outputs
│       └── extinguisher_clean.ply
│
└── benchmark_results/           # Benchmark analysis outputs (generated)
    ├── benchmark_results.csv
    ├── benchmark_summary.txt
    ├── volume_distribution.png
    ├── volume_time_series.png
    ├── volume_histogram.png
    └── volume_correlation.png
```

---

## Output Files

| File | Description | Step |
|------|-------------|------|
| `sparse/0/` | COLMAP sparse reconstruction (cameras, points) | 1 |
| `sparse/scene/extinguisher_raw.ply` | Raw reconstructed point cloud | 1 |
| `sparse/scene/extinguisher_denoised_final.ply` | Denoised point cloud | 2 |
| `sparse/extinguisher/extinguisher_clean.ply` | Cleaned segmented extinguisher | 3 |
| Terminal output | Volume estimation (convex hull, cylinder, average) | 4 |
| `benchmark_results/` | Benchmark analysis outputs (if benchmark run) | - |

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

**Issue: "COLMAP reconstruction gives 0 points or very few points"**
- The reconstruction sometimes fails randomly - simply restart it
- For benchmark pipeline: set `MIN_POINTS_THRESHOLD = 4000` to automatically skip invalid reconstructions
- Check that images have sufficient features and overlap

**Issue: Poor volume estimation**
- Adjust `CYLINDER_RADIUS_FACTOR` in `config.py`
- Tune segmentation parameters (DBSCAN eps, min_points)
- Try different denoising methods (1-3)
- Use the **average volume** (mean of convex hull and cylinder) for best results

**Issue: Python 3.13 installation fails**
- Open3D does not support Python 3.13 yet
- Install Python 3.12, 3.11, 3.10, or 3.9 instead
- Use `py -3.12 -m venv .venv` on Windows to create venv with specific version

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

## Authors

Titouan Pastor
Aurelien Drevet
Aubin Vert
Arthur Melnitchenko 

LTU University

## Acknowledgments

- COLMAP for Structure-from-Motion reconstruction
- Open3D for point cloud processing
- Orbec camera for data capture