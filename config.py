# ==========================================================
#   Configuration Centralisée - Pipeline Extincteur
# ==========================================================

from pathlib import Path

# =========================
# PATHS
# =========================
IMG_DIR = Path("raw/test/camera_color_image_raw")
DB_PATH = Path("colmap/database.db")
SPARSE_DIR = Path("sparse")

# Point clouds
RAW_CLOUD = SPARSE_DIR / "scene/extinguisher_raw.ply"
DENOISED_CLOUD = SPARSE_DIR / "scene/extinguisher_denoised.ply"
DENOISED_FINAL_CLOUD = SPARSE_DIR / "scene/extinguisher_denoised_final.ply"
SEGMENTED_CLOUD = SPARSE_DIR / "scene/extinguisher_segment.ply"
CLEAN_CLOUD = SPARSE_DIR / "extinguisher/extinguisher_clean.ply"

# =========================
# 1. RECONSTRUCTION (COLMAP)
# =========================
MATCHING_METHOD = "EXHAUSTIVE"  # SEQUENTIAL / EXHAUSTIVE

# Camera intrinsics
CAMERA_FX = 306.0
CAMERA_FY = 306.1
CAMERA_CX = 318.5
CAMERA_CY = 201.4
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 400

# SIFT extraction
SIFT_USE_GPU = True
SIFT_MAX_IMAGE_SIZE = 3200
SIFT_PEAK_THRESHOLD = 0.02
SIFT_EDGE_THRESHOLD = 10

# SIFT matching
SIFT_MATCH_USE_GPU = True

# =========================
# 2. DENOISING
# =========================
DENOISING_METHOD = 3  # 1=SOR, 2=ROR, 3=COMBO, 4=DBSCAN

# Méthode 1: Statistical Outlier Removal (SOR)
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Méthode 2: Radius Outlier Removal (ROR)
ROR_NB_POINTS = 16
ROR_RADIUS = 0.05

# Méthode 3: Combinaison SOR + ROR
COMBO_SOR_NEIGHBORS = 50
COMBO_SOR_STD = 2.5
COMBO_ROR_POINTS = 10
COMBO_ROR_RADIUS = 0.3

# Méthode 4: DBSCAN Clustering
DBSCAN_EPS = 0.05
DBSCAN_MIN_POINTS = 10
DBSCAN_KEEP_TOP_N = 1

# =========================
# 3. SEGMENTATION
# =========================
# Downsampling
VOXEL_SIZE = 0.001  # 1 mm

# HSV color filtering (rouge)
RED_H_LOW_MAX = 0.05
RED_H_HIGH_MIN = 0.96
RED_S_MIN = 0.10
RED_V_MIN = 0.10

# DBSCAN clustering
SEG_DBSCAN_EPS = 0.02
SEG_DBSCAN_MIN_P = 40

# Cleaning (SOR + ROR)
SEG_SOR_NEIGHB = 5
SEG_SOR_STD = 1.5
SEG_ROR_POINTS = 10
SEG_ROR_RADIUS = 0.3

# =========================
# 4. VOLUME ESTIMATION
# =========================
# Cylinder adjustment
CYLINDER_RADIUS_FACTOR = 0.9
CYLINDER_HEIGHT_MARGIN_TOP = 0.05
CYLINDER_HEIGHT_MARGIN_BOTTOM = 0.05

# Target volume **
TARGET_MIN = 60 * 0.7  # 42 L
TARGET_MAX = 65 * 1.3  # 84.5 L

# =========================
# VISUALIZATION
# =========================
VISUALIZE = True  # Display Open3D visualizations
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
