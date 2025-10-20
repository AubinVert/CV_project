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
    """Combinaison SOR + ROR"""
    print(f"[COMBO] SOR puis ROR")
    
    # Première passe: SOR
    pcd_temp, ind1 = pcd.remove_statistical_outlier(
        nb_neighbors=sor_neighbors,
        std_ratio=sor_std
    )
    removed_sor = len(pcd.points) - len(pcd_temp.points)
    print(f"  SOR: -{removed_sor} pts")
    
    # Deuxième passe: ROR
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

def method_4_dbscan(pcd, eps, min_points, keep_top_n=1):
    """DBSCAN Clustering - garde les plus gros clusters, vire le bruit"""
    print(f"[DBSCAN] eps={eps}m, min_points={min_points}, keep_top_{keep_top_n}")
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    noise_count = np.sum(labels == -1)
    num_clusters = labels.max() + 1
    
    print(f"  → {num_clusters} clusters trouvés, {noise_count} points de bruit")
    
    if num_clusters == 0:
        print("  [ATTENTION] Aucun cluster trouvé ! Relâche les paramètres.")
        return pcd, np.arange(len(pcd.points))
    
    cluster_sizes = []
    for i in range(num_clusters):
        size = np.sum(labels == i)
        cluster_sizes.append((i, size))
    
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    keep_n = min(keep_top_n, len(cluster_sizes))
    selected_clusters = [cluster_sizes[i][0] for i in range(keep_n)]
    
    print(f"  Clusters gardés: {selected_clusters} (tailles: {[cluster_sizes[i][1] for i in range(keep_n)]})")
    
    mask = np.zeros(len(labels), dtype=bool)
    for cluster_id in selected_clusters:
        mask |= (labels == cluster_id)
    
    indices = np.where(mask)[0]
    pcd_clean = pcd.select_by_index(indices)
    
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Retiré: {removed} pts ({percent:.1f}%)")
    
    return pcd_clean, indices

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
    
    # Appliquer la méthode choisie
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
    elif method == 4:
        pcd_final, _ = method_4_dbscan(
            pcd_original, 
            config.DBSCAN_EPS, 
            config.DBSCAN_MIN_POINTS, 
            config.DBSCAN_KEEP_TOP_N
        )
    else:
        raise ValueError(f"Méthode invalide: {method}")
    
    print(f"Denoised: {len(pcd_final.points)} points")
    
    # Save
    o3d.io.write_point_cloud(str(output_file), pcd_final)
    print(f"[Saved] {output_file}")
    
    # Visualization
    if visualize:
        print("[Visualization]")
        pcd_final.paint_uniform_color([1.0, 0.3, 0.3])
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
