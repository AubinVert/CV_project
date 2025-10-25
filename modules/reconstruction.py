# ==========================================================
#   Module 1: COLMAP Reconstruction
# ==========================================================

import pycolmap
from pathlib import Path
import config

def run_reconstruction():
    """
    Run complete COLMAP reconstruction.
    
    Returns:
        Path: Path to generated raw point cloud
    """
    print("\n" + "="*70)
    print("MODULE 1: COLMAP RECONSTRUCTION")
    print("="*70)
    
    # Create directories
    config.SPARSE_DIR.mkdir(exist_ok=True, parents=True)
    
    # Remove old database
    if config.DB_PATH.exists():
        print(f">> Removing old database: {config.DB_PATH}")
        config.DB_PATH.unlink()
    
    # =========================
    # CAMERA PARAMETERS
    # =========================
    print(">> Creating camera with known intrinsics...")
    
    db = pycolmap.Database(str(config.DB_PATH))
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        params=[config.CAMERA_FX, config.CAMERA_FY, config.CAMERA_CX, config.CAMERA_CY]
    )
    camera_id = db.write_camera(camera)
    print(f">> Camera ID: {camera_id}")
    
    # Add all images
    for img_path in sorted(config.IMG_DIR.glob("*.png")):
        db.write_image(pycolmap.Image(
            name=img_path.name,
            camera_id=camera_id
        ))
    
    db.close()
    
    # =========================
    # SIFT FEATURES EXTRACTION
    # =========================
    print(">> Extracting SIFT features...")
    
    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.use_gpu = config.SIFT_USE_GPU
    sift_opts.max_image_size = config.SIFT_MAX_IMAGE_SIZE
    sift_opts.peak_threshold = config.SIFT_PEAK_THRESHOLD
    sift_opts.edge_threshold = config.SIFT_EDGE_THRESHOLD
    
    pycolmap.extract_features(
        database_path=str(config.DB_PATH),
        image_path=str(config.IMG_DIR),
        camera_model="PINHOLE",
        sift_options=sift_opts,
        device=pycolmap.Device.auto
    )
    
    # =========================
    # FEATURE MATCHING
    # =========================
    if config.MATCHING_METHOD == "SEQUENTIAL":
        print(">> Sequential matching...")
        pycolmap.match_sequential(database_path=str(config.DB_PATH))
    else:
        print(">> Exhaustive matching...")
        sif_match_opts = pycolmap.SiftMatchingOptions()
        sif_match_opts.use_gpu = config.SIFT_MATCH_USE_GPU
        
        pycolmap.match_exhaustive(
            database_path=str(config.DB_PATH),
            sift_options=sif_match_opts
        )
    
    # =========================
    # SFM RECONSTRUCTION
    # =========================
    print(">> SfM reconstruction...")
    recons = pycolmap.incremental_mapping(
        database_path=str(config.DB_PATH),
        image_path=str(config.IMG_DIR),
        output_path=str(config.SPARSE_DIR)
    )
    
    if len(recons) == 0:
        raise RuntimeError("No reconstruction found!")
    
    rec = list(recons.values())[0]
    print(f">> Reconstruction complete: {len(rec.images)} images, {len(rec.points3D)} 3D points")
    
    # =========================
    # EXPORT PLY
    # =========================
    rec.export_PLY(str(config.RAW_CLOUD))
    print(f">> Raw cloud exported: {config.RAW_CLOUD}")
    
    print("\n[✓] Module 1 complete\n")
    return config.RAW_CLOUD

if __name__ == "__main__":
    run_reconstruction()
