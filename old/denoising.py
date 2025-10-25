# ==========================================================
#   Point Cloud Denoising Script
# ==========================================================
# Multiple denoising methods to clean point clouds
# Test different approaches to find the best one
# ==========================================================

import numpy as np
import open3d as o3d

# ==================== PARAMETERS ====================
INPUT_FILE = "../sparse/scene/extinguisher_raw.ply"  # Your input file
OUTPUT_FILE = "../sparse/old/extinguisher_denoised_final.ply"  # Output file

# Method selection (1, 2, 3, 4, or "all" to test all)
METHOD = 3  # Options: 1, 2, 3, 4, "all"

# ===== METHOD 1: Statistical Outlier Removal (SOR) =====
# Removes isolated points based on statistical distribution
SOR_NB_NEIGHBORS = 20      # Number of neighbors to consider (↑ = stricter)
SOR_STD_RATIO = 2.0        # Standard deviation ratio (↓ = stricter, removes more points)

# ===== METHOD 2: Radius Outlier Removal (ROR) =====
# Removes points that don't have enough neighbors within a given radius
ROR_NB_POINTS = 16         # Minimum number of required neighbors (↑ = stricter)
ROR_RADIUS = 0.05          # Search radius in meters (↓ = stricter)

# ===== METHOD 3: Combined SOR + ROR =====
# Applies SOR first, then ROR for progressive cleaning
COMBO_SOR_NEIGHBORS = 50
COMBO_SOR_STD = 2.5        # More permissive for first pass
COMBO_ROR_POINTS = 10
COMBO_ROR_RADIUS = 0.3

# ===== METHOD 4: DBSCAN Clustering =====
# Keeps only main clusters, removes isolated points (noise)
DBSCAN_EPS = 0.05          # Max distance between 2 points in same cluster (↓ = stricter)
DBSCAN_MIN_POINTS = 10     # Min points to form a cluster (↑ = stricter)
DBSCAN_KEEP_TOP_N = 1      # Keep the N largest clusters

# ==================== FUNCTIONS ====================

def print_stats(pcd, name="Point Cloud"):
    """Display point cloud statistics."""
    points = np.asarray(pcd.points)
    print(f"{name}: {len(points)} points")

def method_1_sor(pcd, nb_neighbors, std_ratio):
    """Statistical Outlier Removal (SOR)"""
    print(f"[SOR] nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Removed: {removed} pts ({percent:.1f}%)")
    
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
    print(f"  → Removed: {removed} pts ({percent:.1f}%)")
    
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

def method_4_dbscan(pcd, eps, min_points, keep_top_n=1):
    """DBSCAN Clustering - keeps largest clusters, removes noise"""
    print(f"[DBSCAN] eps={eps}m, min_points={min_points}, keep_top_{keep_top_n}")
    
    # DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # -1 = noise, 0+ = clusters
    noise_count = np.sum(labels == -1)
    num_clusters = labels.max() + 1
    
    print(f"  → {num_clusters} clusters found, {noise_count} noise points")
    
    if num_clusters == 0:
        print("  [WARNING] No clusters found! Relax parameters.")
        return pcd, np.arange(len(pcd.points))
    
    # Count points per cluster
    cluster_sizes = []
    for i in range(num_clusters):
        size = np.sum(labels == i)
        cluster_sizes.append((i, size))
    
    # Sort by decreasing size
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top N largest
    keep_n = min(keep_top_n, len(cluster_sizes))
    selected_clusters = [cluster_sizes[i][0] for i in range(keep_n)]
    
    print(f"  Clusters kept: {selected_clusters} (sizes: {[cluster_sizes[i][1] for i in range(keep_n)]})")
    
    # Create mask
    mask = np.zeros(len(labels), dtype=bool)
    for cluster_id in selected_clusters:
        mask |= (labels == cluster_id)
    
    indices = np.where(mask)[0]
    pcd_clean = pcd.select_by_index(indices)
    
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Removed: {removed} pts ({percent:.1f}%)")
    
    return pcd_clean, indices

def visualize_result(cleaned, title="Result"):
    """Visualize only the cleaned result."""
    print(f"[Visualization] {title}")
    
    # Copy to avoid modifying original
    pcd_clean = o3d.geometry.PointCloud(cleaned)
    
    # Color in red
    # pcd_clean.paint_uniform_color([1.0, 0.3, 0.3])
    
    o3d.visualization.draw_geometries(
        [pcd_clean],
        window_name=title,
        width=1280,
        height=720
    )

# ==================== MAIN ====================

print(f"\n[Loading] {INPUT_FILE}")
pcd_original = o3d.io.read_point_cloud(INPUT_FILE)

if len(pcd_original.points) == 0:
    print("[ERROR] File is empty or could not be loaded!")
    exit(1)

print_stats(pcd_original, "ORIGINAL CLOUD")

# ==================== TEST METHODS ====================

if METHOD == "all":
    print("\n[TEST ALL METHODS]\n")
    
    # Test method 1
    pcd_m1, _ = method_1_sor(pcd_original, SOR_NB_NEIGHBORS, SOR_STD_RATIO)
    print_stats(pcd_m1, "Result")
    
    # Test method 2
    print()
    pcd_m2, _ = method_2_ror(pcd_original, ROR_NB_POINTS, ROR_RADIUS)
    print_stats(pcd_m2, "Result")
    
    # Test method 3
    print()
    pcd_m3, _ = method_3_combo(
        pcd_original, 
        COMBO_SOR_NEIGHBORS, 
        COMBO_SOR_STD,
        COMBO_ROR_POINTS, 
        COMBO_ROR_RADIUS
    )
    print_stats(pcd_m3, "Result")
    
    # Test method 4
    print()
    pcd_m4, _ = method_4_dbscan(pcd_original, DBSCAN_EPS, DBSCAN_MIN_POINTS, DBSCAN_KEEP_TOP_N)
    print_stats(pcd_m4, "Result")
    
    # Save all versions
    o3d.io.write_point_cloud("denoised_method1_sor.ply", pcd_m1)
    o3d.io.write_point_cloud("denoised_method2_ror.ply", pcd_m2)
    o3d.io.write_point_cloud("denoised_method3_combo.ply", pcd_m3)
    o3d.io.write_point_cloud("denoised_method4_dbscan.ply", pcd_m4)
    print("\n[Saved] denoised_method1_sor.ply, denoised_method2_ror.ply, denoised_method3_combo.ply, denoised_method4_dbscan.ply")
    
    # Visualize results
    visualize_result(pcd_m1, "Method 1: SOR")
    visualize_result(pcd_m2, "Method 2: ROR")
    visualize_result(pcd_m3, "Method 3: COMBO")
    visualize_result(pcd_m4, "Method 4: DBSCAN")
    
    # Save best (dbscan) as main file
    pcd_final = pcd_m4
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Saved] {OUTPUT_FILE}")

elif METHOD == 1:
    pcd_final, _ = method_1_sor(pcd_original, SOR_NB_NEIGHBORS, SOR_STD_RATIO)
    print_stats(pcd_final, "Result")
    visualize_result(pcd_final, "SOR")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Saved] {OUTPUT_FILE}")

elif METHOD == 2:
    pcd_final, _ = method_2_ror(pcd_original, ROR_NB_POINTS, ROR_RADIUS)
    print_stats(pcd_final, "Result")
    visualize_result(pcd_final, "ROR")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Saved] {OUTPUT_FILE}")

elif METHOD == 3:
    pcd_final, _ = method_3_combo(
        pcd_original,
        COMBO_SOR_NEIGHBORS,
        COMBO_SOR_STD,
        COMBO_ROR_POINTS,
        COMBO_ROR_RADIUS
    )
    print_stats(pcd_final, "Result")
    visualize_result(pcd_final, "COMBO")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Saved] {OUTPUT_FILE}")

elif METHOD == 4:
    pcd_final, _ = method_4_dbscan(pcd_original, DBSCAN_EPS, DBSCAN_MIN_POINTS, DBSCAN_KEEP_TOP_N)
    print_stats(pcd_final, "Result")
    visualize_result(pcd_final, "DBSCAN")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Saved] {OUTPUT_FILE}")

print("\n[✓] Done!\n")
