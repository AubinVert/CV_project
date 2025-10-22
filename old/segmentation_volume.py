# ==========================================================
#   Extinguisher Segmentation & Volume Estimation (Open3D)
# ==========================================================
# Requirements:
#     pip install open3d numpy
# ==========================================================

import numpy as np
import open3d as o3d

# ==================== SEGMENTATION PARAMETERS ====================
# Goal: Target volume = 60-65 liters (0.060-0.065 m³) ± 30%
# =====================================================================

INPUT_FILE = "../sparse/old/extinguisher_denoised_final.ply"

# ----- 1. DOWNSAMPLING (Point Density Reduction) -----
# Larger VOXEL_SIZE → Fewer points, faster, but loses detail
# Recommended: 0.001-0.005 m (1-5 mm) to balance precision and performance
VOXEL_SIZE = 0.001  # 2 mm - good compromise to keep shape without too much noise

# ----- 2. HSV COLOR FILTERING (Red Detection) -----
# H (Hue): Red = [0-0.03] or [0.97-1.0] on [0,1] scale
# S (Saturation): Higher → More "pure" color (not gray/white)
#   - Too low (e.g. 0.05): Includes pale pink, gray → NOISE
#   - Too high (e.g. 0.50): Excludes faded red → LOSES EXTINGUISHER
# V (Value/Brightness): Eliminates too dark/black areas
#   - Too low (e.g. 0.05): Keeps black shadows → NOISE
#   - Too high (e.g. 0.40): Loses poorly lit areas → INCOMPLETE

# Hue range for red (on [0,1] scale)
# Red = low values (close to 0) OR high values (close to 1)
RED_H_LOW_MAX = 0.05   # Max hue for "low" red (0 to 0.03 = red-orange)
RED_H_HIGH_MIN = 0.96  # Min hue for "high" red (0.97 to 1.0 = red-violet)

RED_S_MIN = 0.10  # Minimum saturation - eliminates gray/white noise
RED_V_MIN = 0.10  # Minimum brightness - eliminates too dark shadows

# ----- 3. DBSCAN CLUSTERING (Grouping Red Points) -----
# DBSCAN_EPS: Search radius (m) to connect points together
#   - Too small (e.g. 0.01): Fragments extinguisher into multiple clusters
#   - Too large (e.g. 0.10): Groups noise with extinguisher
# DBSCAN_MIN_P: Minimum points to form a valid cluster
#   - Too low (e.g. 10): Creates many small parasitic clusters
#   - Too high (e.g. 500): May reject extinguisher if few points
DBSCAN_EPS = 0.02    # 2.5 cm - max distance between neighboring points
DBSCAN_MIN_P = 40     # 50 points minimum per cluster - eliminates small noise

# ----- 4. CLEANING - Statistical Outlier Removal (SOR) -----
# Principle: Calculates average distance of each point to its neighbors
# If a point is too far from its neighbors → it's isolated noise
# SOR_NEIGHB: Number of neighbors to analyze around each point
#   - Too low (e.g. 5): Local detection, may miss global noise
#   - Too high (e.g. 100): More robust but slower
# SOR_STD: Standard deviation threshold to reject a point
#   - Too low (e.g. 0.5): VERY aggressive, removes many points
#   - Too high (e.g. 3.0): Keeps too much noise
SOR_NEIGHB = 5       # Analyze 20 neighbors - good balance
SOR_STD = 1.5         # Reject if > 1 std dev - quite strict

# ----- 5. CLEANING - Radius Outlier Removal (ROR) -----
# Principle: Within a given radius, a point must have X minimum neighbors
# Otherwise it's an isolated point (noise) → removed
# ROR_RADIUS: Search radius (m) around each point
#   - Too small (e.g. 0.01): Only detects very close noise
#   - Too large (e.g. 0.20): Too permissive, keeps distant noise
# ROR_POINTS: Minimum required neighbors in this radius
#   - Too low (e.g. 2): Keeps almost everything
#   - Too high (e.g. 50): May remove extinguisher edges
ROR_RADIUS = 0.3      # 1.5 cm - local neighborhood radius
ROR_POINTS = 10        # 8 neighbors minimum - eliminates isolated points

# ----- 6. FITTED CYLINDER PARAMETERS -----
# Fine control of cylinder to better match real shape
CYLINDER_RADIUS_FACTOR = 0.7   # Factor applied to radius (0.5-1.0)
CYLINDER_HEIGHT_MARGIN_TOP = 0.05    # Margin to remove at top (m)
CYLINDER_HEIGHT_MARGIN_BOTTOM = 0.05  # Margin to remove at bottom (m)
# Example: A real extinguisher often has a handle on top and a base at bottom
# These margins allow excluding these parts from volume calculation

# ======================================================================
# INPUT_FILE   = "colmap-workspace/fused.ply"
# VOXEL_SIZE   = 0.001      # 5 mm downsample
# RED_S_MIN    = 0.10       # HSV saturation threshold
# RED_V_MIN    = 0.10       # HSV brightness threshold
# DBSCAN_EPS   = 0.2       # cluster radius (m)
# DBSCAN_MIN_P = 100         # min pts per cluster
# SOR_NEIGHB   = 100         # neighbors for StatisticalOutlierRemoval
# SOR_STD      = 0.1        # stddev threshold
# ROR_POINTS   = 70         # min neighbors for RadiusOutlierRemoval
# ROR_RADIUS   = 0.1        # radius for RadiusOutlierRemoval (m)


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
    hue_mask = ((h <= RED_H_LOW_MAX) | (h >= RED_H_HIGH_MIN))
    sat_mask = s >= RED_S_MIN
    val_mask = v >= RED_V_MIN
    return hue_mask & sat_mask & val_mask


# ==========================================================
# 1. Load point cloud
# ==========================================================
print("\n=== Step 1: Loading point cloud ===")
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
print("\n=== Step 2: HSV color filtering ===")
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
print("\n=== Step 3: DBSCAN clustering ===")
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
print("\n=== Step 4: Cleaning (SOR + ROR) ===")
pcd_clean, ind = pcd_ext.remove_statistical_outlier(nb_neighbors=SOR_NEIGHB, std_ratio=SOR_STD)
print(f"[INFO] After SOR: {len(pcd_clean.points)} pts kept.")

pcd_clean, ind = pcd_ext.remove_radius_outlier(nb_points=ROR_POINTS, radius=ROR_RADIUS)
print(f"[INFO] After ROR: {len(pcd_clean.points)} pts kept.")
o3d.io.write_point_cloud("extinguisher_clean.ply", pcd_clean)

# ==========================================================
# 5. Volume estimation (Convex Hull + Cylinder)
# ==========================================================
print("\n=== Step 5: Volume estimation ===")
pts = np.asarray(pcd_clean.points)
mins, maxs = pts.min(axis=0), pts.max(axis=0)
dims = maxs - mins
print(f"[DEBUG] Bounding Box Dimensions (X,Y,Z) = {dims}")

# Determine height and diameter
height_raw = np.max(dims)
# Apply top/bottom margins
height_adjusted = height_raw - CYLINDER_HEIGHT_MARGIN_TOP - CYLINDER_HEIGHT_MARGIN_BOTTOM

# For diameter: average of 2 smallest dimensions
sorted_dims = np.sort(dims)
diameter_raw = np.mean(sorted_dims[:2])  # Average of 2 smallest dimensions
radius_raw = diameter_raw / 2.0

# Apply radius factor
radius_adjusted = radius_raw * CYLINDER_RADIUS_FACTOR

print(f"[DEBUG] Height (raw) = {height_raw:.3f} m")
print(f"[DEBUG] Height (adjusted with margins) = {height_adjusted:.3f} m")
print(f"[DEBUG] Diameter (raw) = {diameter_raw:.3f} m")
print(f"[DEBUG] Radius (raw) = {radius_raw:.3f} m")
print(f"[DEBUG] Radius (adjusted x{CYLINDER_RADIUS_FACTOR}) = {radius_adjusted:.3f} m")

# --- METHOD 1: Convex Hull ---
hull, _ = pcd_clean.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color([0, 1, 0])

try:
    vol_hull = hull.get_volume()
except AttributeError:
    verts, tris = np.asarray(hull.vertices), np.asarray(hull.triangles)
    vol_hull = abs(np.sum([np.dot(verts[t[0]], np.cross(verts[t[1]], verts[t[2]])) for t in tris])) / 6.0

# --- METHOD 2: Adjusted cylinder with margins ---
# Cylinder volume = π * r² * h
vol_cylinder_adjusted = np.pi * (radius_adjusted ** 2) * height_adjusted

volume_liters_hull = vol_hull * 1000
volume_liters_cylinder = vol_cylinder_adjusted * 1000

target_min, target_max = 60 * 0.7, 65 * 1.3  # ±30% margin

print(f"\n{'='*70}")
print(f"[RESULTS - 2 METHODS]")
print(f"{'='*70}")
print(f"1. Convex Hull (often underestimates):")
print(f"   → {vol_hull:.4f} m³ = {volume_liters_hull:.1f} L")
print(f"\n2. Adjusted Cylinder (RECOMMENDED):")
print(f"   Radius: {radius_raw:.3f} m × {CYLINDER_RADIUS_FACTOR} = {radius_adjusted:.3f} m")
print(f"   Height: {height_raw:.3f} m - {CYLINDER_HEIGHT_MARGIN_TOP:.3f} m (top) - {CYLINDER_HEIGHT_MARGIN_BOTTOM:.3f} m (bottom) = {height_adjusted:.3f} m")
print(f"   → {vol_cylinder_adjusted:.4f} m³ = {volume_liters_cylinder:.1f} L")
print(f"\n[TARGET] Goal: 60-65 L ± 30% → [{target_min:.1f} - {target_max:.1f}] L")

# Check which method is within target
status_hull = "✓" if target_min <= volume_liters_hull <= target_max else "✗"
status_cyl = "✓" if target_min <= volume_liters_cylinder <= target_max else "✗"

print(f"\n[STATUS] Hull: {status_hull} | Adjusted Cylinder: {status_cyl}")
print(f"{'='*70}")

# ==========================================================
# 5b. Creating cylinder meshes for visualization
# ==========================================================
print("\n=== Step 5b: Creating cylinder meshes ===")

# Calculate bounding box center to position cylinders
center = (mins + maxs) / 2.0

# Determine main axis (one with largest dimension = height)
axis_idx = np.argmax(dims)
print(f"[DEBUG] Main axis (height) = axis {axis_idx} (0=X, 1=Y, 2=Z)")

# --- Adjusted cylinder (in cyan) ---
cylinder_adjusted = o3d.geometry.TriangleMesh.create_cylinder(
    radius=radius_adjusted,
    height=height_adjusted,
    resolution=50,
    split=4
)

# Rotation to align cylinder with main axis
# By default, Open3D creates cylinders along Z axis
# So we must rotate to align with the axis of largest dimension
if axis_idx == 0:  # X axis
    # 90° rotation around Y
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
elif axis_idx == 1:  # Y axis
    # 90° rotation around X
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
else:  # Z axis (axis_idx == 2)
    # No rotation needed
    R = np.eye(3)

# Apply rotation and translation
cylinder_adjusted.rotate(R, center=[0, 0, 0])
cylinder_adjusted.translate(center)
cylinder_adjusted.paint_uniform_color([0, 1, 1])  # Cyan

# Convert to wireframe for better visualization
cylinder_adjusted_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder_adjusted)
cylinder_adjusted_ls.paint_uniform_color([0, 1, 1])  # Cyan

print(f"[OK] Adjusted cylinder created and aligned (cyan)")

# ==========================================================
# 6. Visualization
# ==========================================================
print("\n=== Step 6: Visualization ===")
print("[LEGEND]")
print("  🔴 Red  : Segmented extinguisher points")
print("  🟢 Green   : Convex Hull")
print("  🩵 Cyan   : Adjusted Cylinder (with margins and radius factor)")

try:
    o3d.visualization.draw_geometries(
        [pcd_clean, hull_ls, cylinder_adjusted_ls],
        window_name="Extinguisher - Hull vs Adjusted Cylinder",
        width=1280,
        height=720
    )
except:
    print("[WARN] Visualization skipped (no GUI context).")

print("\n[✓] Pipeline complete.")
