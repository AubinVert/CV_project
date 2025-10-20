# ==========================================================
#   Module 2: Point Cloud Denoising
# ==========================================================

import numpy as np
import open3d as o3d
from pathlib import Path
import config

def method_1_sor(pcd, nb_neighbors, std_ratio):
    """Statistical Outlier Removal (SOR)"""
    print(f"[SOR] nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Retiré: {removed} pts ({percent:.1f}%)")
    return pcd_clean, ind

def method_2_ror(pcd, nb_points, radius):
    """Radius Outlier Removal (ROR)"""
    print(f"[ROR] nb_points={nb_points}, radius={radius}m")
    pcd_clean, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius
    )
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Retiré: {removed} pts ({percent:.1f}%)")
    return pcd_clean, ind

def method_3_combo(pcd, sor_neighbors, sor_std, ror_points, ror_radius):
    """Combined SOR + ROR"""
    print(f"[COMBO] SOR then ROR")
    
    # First pass: SOR
    pcd_temp, ind1 = pcd.remove_statistical_outlier(
        nb_neighbors=sor_neighbors,
        std_ratio=sor_std
    )
    removed_sor = len(pcd.points) - len(pcd_temp.points)
    print(f"  SOR: -{removed_sor} pts")
    
    # Second pass: ROR
    pcd_clean, ind2 = pcd_temp.remove_radius_outlier(
        nb_points=ror_points,
        radius=ror_radius
    )
    removed_ror = len(pcd_temp.points) - len(pcd_clean.points)
    print(f"  ROR: -{removed_ror} pts")
    
    total_removed = len(pcd.points) - len(pcd_clean.points)
    total_percent = (total_removed / len(pcd.points)) * 100
    print(f"  Total: -{total_removed} pts ({total_percent:.1f}%)")
    
    return pcd_clean, ind2

def run_denoising(input_file=None, output_file=None, method=None, visualize=None):
    """
    Run point cloud denoising.
    
    Args:
        input_file: Input file path (default: config.RAW_CLOUD)
        output_file: Output file path (default: config.DENOISED_FINAL_CLOUD)
        method: Denoising method (default: config.DENOISING_METHOD)
        visualize: Display visualization (default: config.VISUALIZE)
    
    Returns:
        Path: Path to denoised cloud
    """
    print("\n" + "="*70)
    print("MODULE 2: DENOISING")
    print("="*70)
    
    # Default values
    input_file = input_file or config.RAW_CLOUD
    output_file = output_file or config.DENOISED_FINAL_CLOUD
    method = method if method is not None else config.DENOISING_METHOD
    visualize = visualize if visualize is not None else config.VISUALIZE
    
    print(f"\n[Loading] {input_file}")
    pcd_original = o3d.io.read_point_cloud(str(input_file))
    
    if len(pcd_original.points) == 0:
        raise RuntimeError("Empty file or loading error!")
    
    print(f"Original cloud: {len(pcd_original.points)} points")
    
    # Apply chosen method
    if method == 1:
        pcd_final, _ = method_1_sor(pcd_original, config.SOR_NB_NEIGHBORS, config.SOR_STD_RATIO)
    elif method == 2:
        pcd_final, _ = method_2_ror(pcd_original, config.ROR_NB_POINTS, config.ROR_RADIUS)
    elif method == 3:
        pcd_final, _ = method_3_combo(
            pcd_original,
            config.COMBO_SOR_NEIGHBORS,
            config.COMBO_SOR_STD,
            config.COMBO_ROR_POINTS,
            config.COMBO_ROR_RADIUS
        )
    else:
        raise ValueError(f"Invalid method: {method}. Valid range: 1-3")
    
    print(f"Denoised: {len(pcd_final.points)} points")
    
    # Save
    o3d.io.write_point_cloud(str(output_file), pcd_final)
    print(f"[Saved] {output_file}")
    
    # Visualization
    if visualize:
        print("[Visualization]")
        o3d.visualization.draw_geometries(
            [pcd_final],
            window_name=f"Step 2/4: Denoised Point Cloud ({len(pcd_final.points)} pts)",
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT
        )
    
    print("\n[✓] Module 2 complete\n")
    return Path(output_file)

if __name__ == "__main__":
    run_denoising()
