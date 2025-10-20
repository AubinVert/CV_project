# ==========================================================
#   Extinguisher Segmentation & Volume Estimation (Open3D)
# ==========================================================
# Requirements:
#     pip install open3d numpy
# ==========================================================

import numpy as np
import open3d as o3d

# ==================== PARAMÈTRES DE SEGMENTATION ====================
# Objectif: Volume cible = 60-65 litres (0.060-0.065 m³) ± 30%
# =====================================================================

INPUT_FILE = "sparse/extinguisher_denoised.ply"

# ----- 1. DOWNSAMPLING (Réduction de la densité de points) -----
# Plus VOXEL_SIZE est grand → Moins de points, plus rapide, mais perd des détails
# Recommandé: 0.001-0.005 m (1-5 mm) pour équilibrer précision et performance
VOXEL_SIZE = 0.001  # 2 mm - bon compromis pour garder la forme sans trop de bruit

# ----- 2. FILTRAGE COULEUR HSV (Détection du rouge) -----
# H (Hue/Teinte): Rouge = [0-0.03] ou [0.97-1.0] sur une échelle [0,1]
# S (Saturation): Plus c'est haut → Plus la couleur est "pure" (pas gris/blanc)
#   - Trop bas (ex: 0.05): Inclut du rose pâle, du gris → BRUIT
#   - Trop haut (ex: 0.50): Exclut du rouge délavé → PERD L'EXTINCTEUR
# V (Value/Luminosité): Élimine les zones trop sombres/noires
#   - Trop bas (ex: 0.05): Garde des ombres noires → BRUIT
#   - Trop haut (ex: 0.40): Perd les zones peu éclairées → INCOMPLET

# Range de Hue pour le rouge (sur échelle [0,1])
# Rouge = basses valeurs (proche de 0) OU hautes valeurs (proche de 1)
RED_H_LOW_MAX = 0.05   # Hue max pour le rouge "bas" (0 à 0.03 = rouge-orange)
RED_H_HIGH_MIN = 0.96  # Hue min pour le rouge "haut" (0.97 à 1.0 = rouge-violet)

RED_S_MIN = 0.10  # Saturation minimale - élimine le bruit gris/blanc
RED_V_MIN = 0.10  # Luminosité minimale - élimine les ombres trop sombres

# ----- 3. CLUSTERING DBSCAN (Regroupement des points rouges) -----
# DBSCAN_EPS: Rayon de recherche (m) pour connecter des points ensemble
#   - Trop petit (ex: 0.01): Fragmente l'extincteur en plusieurs clusters
#   - Trop grand (ex: 0.10): Regroupe le bruit avec l'extincteur
# DBSCAN_MIN_P: Nombre minimum de points pour former un cluster valide
#   - Trop bas (ex: 10): Crée beaucoup de petits clusters parasites
#   - Trop haut (ex: 500): Risque de rejeter l'extincteur si peu de points
DBSCAN_EPS = 0.02    # 2.5 cm - distance max entre points voisins
DBSCAN_MIN_P = 40     # 50 points minimum par cluster - élimine petits bruits

# ----- 4. NETTOYAGE - Statistical Outlier Removal (SOR) -----
# Principe: Calcule la distance moyenne de chaque point à ses voisins
# Si un point est trop loin de ses voisins → c'est du bruit isolé
# SOR_NEIGHB: Nombre de voisins à analyser autour de chaque point
#   - Trop bas (ex: 5): Détection locale, peut manquer du bruit global
#   - Trop haut (ex: 100): Plus robuste mais plus lent
# SOR_STD: Seuil en écart-type pour rejeter un point
#   - Trop bas (ex: 0.5): TRÈS agressif, supprime beaucoup de points
#   - Trop haut (ex: 3.0): Garde trop de bruit
SOR_NEIGHB = 5       # Analyse 20 voisins - bon équilibre
SOR_STD = 1.5         # Rejette si > 1 écart-type - assez strict

# ----- 5. NETTOYAGE - Radius Outlier Removal (ROR) -----
# Principe: Dans un rayon donné, un point doit avoir X voisins minimum
# Sinon c'est un point isolé (bruit) → supprimé
# ROR_RADIUS: Rayon de recherche (m) autour de chaque point
#   - Trop petit (ex: 0.01): Ne détecte que le bruit très proche
#   - Trop grand (ex: 0.20): Trop permissif, garde du bruit lointain
# ROR_POINTS: Nombre minimum de voisins requis dans ce rayon
#   - Trop bas (ex: 2): Garde presque tout
#   - Trop haut (ex: 50): Peut supprimer des bords de l'extincteur
ROR_RADIUS = 0.3      # 1.5 cm - rayon de voisinage local
ROR_POINTS = 10        # 8 voisins minimum - élimine points isolés

# ----- 6. PARAMÈTRES DU CYLINDRE AJUSTÉ -----
# Contrôle fin du cylindre pour mieux correspondre à la forme réelle
CYLINDER_RADIUS_FACTOR = 0.7   # Facteur appliqué au rayon (0.5-1.0)
CYLINDER_HEIGHT_MARGIN_TOP = 0.05    # Marge à retirer en haut (m)
CYLINDER_HEIGHT_MARGIN_BOTTOM = 0.05  # Marge à retirer en bas (m)
# Exemple: Un extincteur réel a souvent une poignée en haut et un pied en bas
# Ces marges permettent d'exclure ces parties du calcul de volume

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
# 5. Volume estimation (Convex Hull + Cylindre)
# ==========================================================
print("\n=== Step 5 : Volume estimation ===")
pts = np.asarray(pcd_clean.points)
mins, maxs = pts.min(axis=0), pts.max(axis=0)
dims = maxs - mins
print(f"[DEBUG] Bounding Box Dimensions (X,Y,Z) = {dims}")

# Déterminer hauteur et diamètre
height_raw = np.max(dims)
# Appliquer les marges haut/bas
height_adjusted = height_raw - CYLINDER_HEIGHT_MARGIN_TOP - CYLINDER_HEIGHT_MARGIN_BOTTOM

# Pour le diamètre: moyenne des 2 plus petites dimensions
sorted_dims = np.sort(dims)
diameter_raw = np.mean(sorted_dims[:2])  # Moyenne des 2 plus petites dimensions
radius_raw = diameter_raw / 2.0

# Appliquer le facteur de rayon
radius_adjusted = radius_raw * CYLINDER_RADIUS_FACTOR

print(f"[DEBUG] Height (brute) = {height_raw:.3f} m")
print(f"[DEBUG] Height (ajustée avec marges) = {height_adjusted:.3f} m")
print(f"[DEBUG] Diameter (brut) = {diameter_raw:.3f} m")
print(f"[DEBUG] Radius (brut) = {radius_raw:.3f} m")
print(f"[DEBUG] Radius (ajusté x{CYLINDER_RADIUS_FACTOR}) = {radius_adjusted:.3f} m")

# --- MÉTHODE 1: Convex Hull ---
hull, _ = pcd_clean.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color([0, 1, 0])

try:
    vol_hull = hull.get_volume()
except AttributeError:
    verts, tris = np.asarray(hull.vertices), np.asarray(hull.triangles)
    vol_hull = abs(np.sum([np.dot(verts[t[0]], np.cross(verts[t[1]], verts[t[2]])) for t in tris])) / 6.0

# --- MÉTHODE 2: Cylindre ajusté avec marges ---
# Volume cylindre = π * r² * h
vol_cylinder_adjusted = np.pi * (radius_adjusted ** 2) * height_adjusted

volume_liters_hull = vol_hull * 1000
volume_liters_cylinder = vol_cylinder_adjusted * 1000

target_min, target_max = 60 * 0.7, 65 * 1.3  # ±30% de marge

print(f"\n{'='*70}")
print(f"[RÉSULTATS - 2 MÉTHODES]")
print(f"{'='*70}")
print(f"1. Convex Hull (sous-estime souvent):")
print(f"   → {vol_hull:.4f} m³ = {volume_liters_hull:.1f} L")
print(f"\n2. Cylindre ajusté (RECOMMANDÉ):")
print(f"   Rayon: {radius_raw:.3f} m × {CYLINDER_RADIUS_FACTOR} = {radius_adjusted:.3f} m")
print(f"   Hauteur: {height_raw:.3f} m - {CYLINDER_HEIGHT_MARGIN_TOP:.3f} m (haut) - {CYLINDER_HEIGHT_MARGIN_BOTTOM:.3f} m (bas) = {height_adjusted:.3f} m")
print(f"   → {vol_cylinder_adjusted:.4f} m³ = {volume_liters_cylinder:.1f} L")
print(f"\n[TARGET] Objectif: 60-65 L ± 30% → [{target_min:.1f} - {target_max:.1f}] L")

# Vérifier quelle méthode est dans la cible
status_hull = "✓" if target_min <= volume_liters_hull <= target_max else "✗"
status_cyl = "✓" if target_min <= volume_liters_cylinder <= target_max else "✗"

print(f"\n[STATUS] Hull: {status_hull} | Cylindre ajusté: {status_cyl}")
print(f"{'='*70}")

# ==========================================================
# 5b. Création des meshes cylindriques pour visualisation
# ==========================================================
print("\n=== Step 5b : Creating cylinder meshes ===")

# Calcul du centre de la bounding box pour positionner les cylindres
center = (mins + maxs) / 2.0

# Déterminer l'axe principal (celui avec la plus grande dimension = hauteur)
axis_idx = np.argmax(dims)
print(f"[DEBUG] Axe principal (hauteur) = axe {axis_idx} (0=X, 1=Y, 2=Z)")

# --- Cylindre ajusté (en cyan) ---
cylinder_adjusted = o3d.geometry.TriangleMesh.create_cylinder(
    radius=radius_adjusted,
    height=height_adjusted,
    resolution=50,
    split=4
)

# Rotation pour aligner le cylindre avec l'axe principal
# Par défaut, Open3D crée les cylindres selon l'axe Z
# On doit donc pivoter pour aligner avec l'axe de plus grande dimension
if axis_idx == 0:  # Axe X
    # Rotation de 90° autour de Y
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
elif axis_idx == 1:  # Axe Y
    # Rotation de 90° autour de X
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
else:  # Axe Z (axis_idx == 2)
    # Pas de rotation nécessaire
    R = np.eye(3)

# Appliquer rotation et translation
cylinder_adjusted.rotate(R, center=[0, 0, 0])
cylinder_adjusted.translate(center)
cylinder_adjusted.paint_uniform_color([0, 1, 1])  # Cyan

# Conversion en wireframe pour meilleure visualisation
cylinder_adjusted_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder_adjusted)
cylinder_adjusted_ls.paint_uniform_color([0, 1, 1])  # Cyan

print(f"[OK] Cylindre ajusté créé et aligné (cyan)")

# ==========================================================
# 6. Visualization
# ==========================================================
print("\n=== Step 6 : Visualization ===")
print("[LÉGENDE]")
print("  🔴 Rouge  : Points segmentés de l'extincteur")
print("  🟢 Vert   : Convex Hull")
print("  🩵 Cyan   : Cylindre ajusté (avec marges et facteur de rayon)")

try:
    o3d.visualization.draw_geometries(
        [pcd_clean, hull_ls, cylinder_adjusted_ls],
        window_name="Extinguisher - Hull vs Cylindre Ajusté",
        width=1280,
        height=720
    )
except:
    print("[WARN] Visualization skipped (no GUI context).")

print("\n[✓] Pipeline complete.")
