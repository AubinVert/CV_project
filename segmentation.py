# ==========================================================
#   Extinguisher Segmentation & Volume Estimation (Open3D)
# ==========================================================
# Requirements:
#     pip install open3d numpy
# ==========================================================

import numpy as np
import open3d as o3d

# -------------------- PARAMETERS ---------------------------
INPUT_FILE   = "parseOut.ply"
VOXEL_SIZE   = 0.005      # 5 mm downsample
RED_S_MIN    = 0.10       # HSV saturation threshold
RED_V_MIN    = 0.10       # HSV brightness threshold
DBSCAN_EPS   = 0.4       # cluster radius (m)
DBSCAN_MIN_P = 40         # min pts per cluster
SOR_NEIGHB   = 5         # neighbors for StatisticalOutlierRemoval
SOR_STD      = 1.5        # stddev threshold
ROR_POINTS   = 5         # min neighbors for RadiusOutlierRemoval
ROR_RADIUS   = 0.5        # radius for RadiusOutlierRemoval (m)
# ------------------------------------------------------------


# ==========================================================
# Helper functions
# ==========================================================
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
    hue_mask = ((h <= 0.01) | (h >= 0.99))
    sat_mask = s >= RED_S_MIN
    val_mask = v >= RED_V_MIN
    return hue_mask & sat_mask & val_mask


# ==========================================================
# 1. Load point cloud
# ==========================================================
print("\n=== Step 1 : Loading point cloud ===")
pcd = o3d.io.read_point_cloud(INPUT_FILE)
if pcd.is_empty():
    raise RuntimeError("Failed to load parseOut.ply or it's empty.")
print(f"[INFO] Loaded {len(pcd.points)} points from '{INPUT_FILE}'.")

# Optional downsample
# pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
pcd.estimate_normals()
print(f"[INFO] After downsampling: {len(pcd.points)} points.")

# ==========================================================
# 2. Color-based segmentation (Red detection)
# ==========================================================
print("\n=== Step 2 : HSV color filtering ===")
if not pcd.has_colors():
    print("[WARN] No colors in file — skipping HSV filtering.")
    red_mask = np.ones(len(pcd.points), dtype=bool)
else:
    colors = np.asarray(pcd.colors).clip(0, 1)
    hsv = rgb_to_hsv(colors)
    red_mask = keep_red_points(hsv)

pcd_red = pcd.select_by_index(np.where(red_mask)[0])
# show red pcd
o3d.visualization.draw_geometries([pcd_red],
                                  window_name="red pcd")
pcd_non_red = pcd.select_by_index(np.where(~red_mask)[0])
print(f"[INFO] Red-candidate points: {len(pcd_red.points)}")

if pcd_red.is_empty():
    raise RuntimeError("No red points detected. Try loosening thresholds.")

# ==========================================================
# 3. Clustering (keep largest red cluster)
# ==========================================================
print("\n=== Step 3 : DBSCAN clustering ===")
labels = np.array(pcd_red.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_P))
if labels.size == 0 or np.all(labels < 0):
    labels = np.array(pcd_red.cluster_dbscan(eps=0.03, min_points=20))

n_clusters = labels.max() + 1 if labels.size else 0
print(f"[INFO] Found {n_clusters} clusters.")

if n_clusters <= 0:
    extinguisher_idx = np.arange(len(pcd_red.points))
else:
    counts = [np.sum(labels == k) for k in range(n_clusters)]
    extinguisher_label = int(np.argmax(counts))
    extinguisher_idx = np.where(labels == extinguisher_label)[0]
    print(f"[INFO] Largest cluster: #{extinguisher_label} ({counts[extinguisher_label]} pts)")

pcd_ext = pcd_red.select_by_index(extinguisher_idx)
pcd_ext.paint_uniform_color([1, 0, 0])

# Save raw segment
o3d.io.write_point_cloud("extinguisher_segment.ply", pcd_ext)
print("[OK] Saved 'extinguisher_segment.ply'.")

# ==========================================================
# 4. Cleaning (Outlier removal)
# ==========================================================
print("\n=== Step 4 : Cleaning (SOR + ROR) ===")
pcd_clean, ind = pcd_ext.remove_statistical_outlier(nb_neighbors=SOR_NEIGHB, std_ratio=SOR_STD)
print(f"[INFO] After SOR: {len(pcd_clean.points)} pts kept.")

pcd_clean, ind = pcd_ext.remove_radius_outlier(nb_points=ROR_POINTS, radius=ROR_RADIUS)
print(f"[INFO] After ROR: {len(pcd_clean.points)} pts kept.")
o3d.io.write_point_cloud("extinguisher_clean.ply", pcd_clean)

# ==========================================================
# 5. Volume estimation (Convex Hull)
# ==========================================================
print("\n=== Step 5 : Volume estimation ===")
pts = np.asarray(pcd_clean.points)
mins, maxs = pts.min(axis=0), pts.max(axis=0)
dims = maxs - mins
print(f"[DEBUG] Dimensions (X,Y,Z) = {dims}")
print(f"[DEBUG] Approx. height = {np.max(dims):.3f} m")

hull, _ = pcd_clean.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color([0, 1, 0])

try:
    vol = hull.get_volume()
except AttributeError:
    verts, tris = np.asarray(hull.vertices), np.asarray(hull.triangles)
    vol = abs(np.sum([np.dot(verts[t[0]], np.cross(verts[t[1]], verts[t[2]])) for t in tris])) / 6.0

print(f"[RESULT] Estimated volume (Convex Hull): {vol:.4f} m³  ≈ {vol*1000:.1f} L")

# ==========================================================
# 6. Visualization
# ==========================================================
print("\n=== Step 6 : Visualization ===")
try:
    o3d.visualization.draw_geometries([pcd_clean, hull_ls],
                                      window_name="Extinguisher + Convex Hull")
except:
    print("[WARN] Visualization skipped (no GUI context).")

print("\n[✓] Pipeline complete.")
