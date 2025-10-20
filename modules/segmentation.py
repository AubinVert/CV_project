# ==========================================================
#   Module 3: Red Extinguisher Segmentation
# ==========================================================

import numpy as np
import open3d as o3d
from pathlib import Path
import config

def rgb_to_hsv(rgb):
    """rgb: Nx3 in [0,1] -> hsv in [0,1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    diff = mx - mn + 1e-8
    h = np.zeros_like(mx)
    mask = (mx == r)
    h[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6
    mask = (mx == g)
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2
    mask = (mx == b)
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4
    h = (h / 6.0) % 1.0
    s = diff / (mx + 1e-8)
    v = mx
    return np.stack([h, s, v], axis=1)

def keep_red_points(colors_hsv):
    """Return boolean mask for 'red' points in HSV."""
    h, s, v = colors_hsv[:, 0], colors_hsv[:, 1], colors_hsv[:, 2]
    hue_mask = ((h <= config.RED_H_LOW_MAX) | (h >= config.RED_H_HIGH_MIN))
    sat_mask = s >= config.RED_S_MIN
    val_mask = v >= config.RED_V_MIN
    return hue_mask & sat_mask & val_mask

def run_segmentation(input_file=None, output_file=None, visualize=None):
    """
    Segment red extinguisher from point cloud.
    
    Args:
        input_file: Input file path (default: config.DENOISED_FINAL_CLOUD)
        output_file: Output file path (default: config.CLEAN_CLOUD)
        visualize: Display visualizations (default: config.VISUALIZE)
    
    Returns:
        Path: Path to segmented and cleaned cloud
    """
    print("\n" + "="*70)
    print("MODULE 3: SEGMENTATION")
    print("="*70)
    
    # Default values
    input_file = input_file or config.DENOISED_FINAL_CLOUD
    output_file = output_file or config.CLEAN_CLOUD
    visualize = visualize if visualize is not None else config.VISUALIZE
    
    # ==========================================================
    # 1. Load point cloud
    # ==========================================================
    print("\n=== Step 1: Load ===")
    pcd = o3d.io.read_point_cloud(str(input_file))
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load {input_file}")
    print(f"Loaded: {len(pcd.points)} points")
    
    # Optional downsample
    # pcd = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE)
    pcd.estimate_normals()
    
    # ==========================================================
    # 2. Color-based segmentation (Red detection)
    # ==========================================================
    print("\n=== Step 2: HSV Color Filtering ===")
    if not pcd.has_colors():
        print("[WARN] No colors — skipping HSV filtering")
        red_mask = np.ones(len(pcd.points), dtype=bool)
    else:
        colors = np.asarray(pcd.colors).clip(0, 1)
        hsv = rgb_to_hsv(colors)
        red_mask = keep_red_points(hsv)
    
    pcd_red = pcd.select_by_index(np.where(red_mask)[0])
    
    # Visualization: red points
    if visualize:
        print("[Visualization] Red points detected")
        o3d.visualization.draw_geometries(
            [pcd_red],
            window_name=f"Step 3/4: Red Points Detection ({len(pcd_red.points)} pts)",
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT
        )
    
    print(f"Red candidates: {len(pcd_red.points)} points")
    
    if pcd_red.is_empty():
        raise RuntimeError("No red points detected")
    
    # ==========================================================
    # 3. Clustering (keep largest red cluster)
    # ==========================================================
    print("\n=== Step 3: DBSCAN Clustering ===")
    labels = np.array(pcd_red.cluster_dbscan(eps=config.SEG_DBSCAN_EPS, min_points=config.SEG_DBSCAN_MIN_P))
    if labels.size == 0 or np.all(labels < 0):
        labels = np.array(pcd_red.cluster_dbscan(eps=0.03, min_points=20))
    
    n_clusters = labels.max() + 1 if labels.size else 0
    print(f"Found {n_clusters} clusters")
    
    if n_clusters <= 0:
        extinguisher_idx = np.arange(len(pcd_red.points))
    else:
        counts = [np.sum(labels == k) for k in range(n_clusters)]
        extinguisher_label = int(np.argmax(counts))
        extinguisher_idx = np.where(labels == extinguisher_label)[0]
        print(f"Largest cluster: #{extinguisher_label} ({counts[extinguisher_label]} pts)")
    
    pcd_ext = pcd_red.select_by_index(extinguisher_idx)
    pcd_ext.paint_uniform_color([1, 0, 0])
    
    # Save raw segment
    o3d.io.write_point_cloud(str(config.SEGMENTED_CLOUD), pcd_ext)
    print(f"[Saved] {config.SEGMENTED_CLOUD}")
    
    # ==========================================================
    # 4. Cleaning (Outlier removal)
    # ==========================================================
    print("\n=== Step 4: Cleaning (SOR + ROR) ===")
    pcd_clean, ind = pcd_ext.remove_statistical_outlier(
        nb_neighbors=config.SEG_SOR_NEIGHB, 
        std_ratio=config.SEG_SOR_STD
    )
    print(f"After SOR: {len(pcd_clean.points)} pts")
    
    pcd_clean, ind = pcd_clean.remove_radius_outlier(
        nb_points=config.SEG_ROR_POINTS, 
        radius=config.SEG_ROR_RADIUS
    )
    print(f"After ROR: {len(pcd_clean.points)} pts")
    
    o3d.io.write_point_cloud(str(output_file), pcd_clean)
    print(f"[Saved] {output_file}")
    
    # Final visualization
    if visualize:
        print("[Visualization] Segmented extinguisher")
        pcd_clean.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd_clean],
            window_name=f"Step 3/4: Segmented Extinguisher ({len(pcd_clean.points)} pts)",
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT
        )
    
    print("\n[✓] Module 3 complete\n")
    return Path(output_file)

if __name__ == "__main__":
    run_segmentation()
