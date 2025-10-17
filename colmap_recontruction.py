import pycolmap
from pathlib import Path
import open3d as o3d

# =========================
# CONFIGURATION
# =========================
IMG_DIR = Path("raw/test/camera_color_image_raw")
DB_PATH = Path("database.db")
OUT_DIR = Path("sparse")
OUT_DIR.mkdir(exist_ok=True, parents=True)
MATCHING_METHOD = "EXHAUSTIVE" # SEQUENTIAL / EXHAUSTIVE

# =========================
# OLD BASE DELETION
# =========================
if DB_PATH.exists():
    print(f">> Old base deletion: {DB_PATH}")
    DB_PATH.unlink()

# =========================
# CAMERA PARAMETERS
# =========================
print(">> Creating camera with known intrinsics...")

# camera parameters
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
width, height = 640, 400

# Adding camera in database
db = pycolmap.Database(str(DB_PATH))
camera = pycolmap.Camera(
    model="PINHOLE",
    width=width,
    height=height,
    params=[fx, fy, cx, cy]
)
camera_id = db.write_camera(camera)
print(f">> Camera ID: {camera_id}")

# Add all images using this same camera
for img_path in sorted(IMG_DIR.glob("*.png")):
    db.write_image(pycolmap.Image(
        name=img_path.name,
        camera_id=camera_id
    ))


db.close()

# =========================
# SIFT FEATURES EXTRACTION
# =========================
print(">> SIFT features extraction …")

sift_opts = pycolmap.SiftExtractionOptions()
sift_opts.use_gpu = True
sift_opts.max_image_size = 3200
sift_opts.peak_threshold = 0.02
sift_opts.edge_threshold = 10

pycolmap.extract_features(
    database_path=str(DB_PATH),
    image_path=str(IMG_DIR),
    camera_model="PINHOLE",
    #camera_mode=pycolmap.CameraMode.AUTO,
    sift_options=sift_opts,
    device=pycolmap.Device.auto
)

# =========================
# FEATURE MATCHING
# =========================

if MATCHING_METHOD == "SEQUENTIAL":
    print(">> Sequential feature matching…")

    pycolmap.match_sequential(
        database_path=str(DB_PATH)
    )
else:
    print(">> Exhaustive feature matching…")
    sif_match_opts = pycolmap.SiftMatchingOptions()
    sif_match_opts.use_gpu = True

    pycolmap.match_exhaustive(
        database_path=str(DB_PATH),
        sift_options=sif_match_opts
    )

# =========================
# SFM RECONSTRUCTION
# =========================
print(">> SfM reconstruction…")
recons = pycolmap.incremental_mapping(
    database_path=str(DB_PATH),
    image_path=str(IMG_DIR),
    output_path=str(OUT_DIR)
)

if len(recons) == 0:
    raise RuntimeError("No reconstruction found !")

rec = list(recons.values())[0]
print(f">> Reconstruction is finished : {len(rec.images)} images, {len(rec.points3D)} points 3D.")

# =========================
# EXPORT PLY
# =========================
ply_path = OUT_DIR / "extinguisher_raw.ply"
rec.export_PLY(str(ply_path))
print(f">> Exported raw point cloud : {ply_path}")

# =========================
# POST-PROCESSING / DENOISING
# =========================
print(">> Post-processing and denoising with Open3D…")
pcd = o3d.io.read_point_cloud(str(ply_path))

if MATCHING_METHOD == "SEQUENTIAL":
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.3)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
else:
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=60, std_ratio=0.2)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

# =========================
# EXPORT DENOISED PLY
# =========================
denoised_path = OUT_DIR / "extinguisher_denoised.ply"
o3d.io.write_point_cloud(str(denoised_path), pcd)
print(f">> Denoised point cloud exported : {denoised_path}")

# =========================
# Vizualisation OPEN3D
# =========================
print(">> 3D visualization of denoised cloud…")
o3d.visualization.draw_geometries([pcd])