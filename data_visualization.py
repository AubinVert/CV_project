import pycolmap
from pathlib import Path
import open3d as o3d

# =========================
# CONFIGURATION
# =========================
IMG_DIR = Path("raw/test/camera_color_image_raw")  # dossier des images
DB_PATH = Path("database.db")
OUT_DIR = Path("sparse")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Paramètres caméra connus
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
width, height = 640, 400

# =========================
# SUPPRESSION ANCIENNE BASE
# =========================
if DB_PATH.exists():
    print(f">> Suppression ancienne base {DB_PATH}")
    DB_PATH.unlink()

# =========================
# EXTRACTION FEATURES SIFT
# PyCOLMAP crée automatiquement la base SQLite
# =========================
print(">> Extraction des features SIFT…")
from pycolmap import SiftExtractionOptions, Device, CameraMode

sift_opts = SiftExtractionOptions()
sift_opts.use_gpu = True
sift_opts.max_image_size = 3200
sift_opts.peak_threshold = 0.02
sift_opts.edge_threshold = 10

pycolmap.extract_features(
    database_path=str(DB_PATH),
    image_path=str(IMG_DIR),
    camera_model="PINHOLE",
    camera_mode=CameraMode.AUTO,
    sift_options=sift_opts,
    device=Device.auto
)

# =========================
# MATCHING EXHAUSTIF
# =========================
print(">> Matching exhaustif…")
pycolmap.match_exhaustive(
    database_path=str(DB_PATH)
)

# =========================
# RECONSTRUCTION SFM
# =========================
print(">> Reconstruction SfM…")
recons = pycolmap.incremental_mapping(
    database_path=str(DB_PATH),
    image_path=str(IMG_DIR),
    output_path=str(OUT_DIR)
)

if len(recons) == 0:
    raise RuntimeError("❌ Aucune reconstruction trouvée !")

rec = list(recons.values())[0]
print(f">> Reconstruction terminée : {len(rec.images)} images, {len(rec.points3D)} points 3D.")

# =========================
# EXPORT PLY
# =========================
ply_path = OUT_DIR / "sparse_points.ply"
rec.export_PLY(str(ply_path))
print(f">> Nuage de points exporté : {ply_path}")

# =========================
# VISUALISATION OPEN3D
# =========================
pcd = o3d.io.read_point_cloud(str(ply_path))
print(">> Affichage du nuage de points…")
o3d.visualization.draw_geometries([pcd])