# ==========================================================
#   Point Cloud Denoising Script
# ==========================================================
# Plusieurs méthodes de denoising pour nettoyer les nuages de points
# Test différentes approches pour trouver la meilleure
# ==========================================================

import numpy as np
import open3d as o3d

# ==================== PARAMÈTRES ====================
INPUT_FILE = "sparse/extinguisher_denoised.ply"  # Ton fichier d'entrée
OUTPUT_FILE = "sparse/extinguisher_denoised_final.ply"  # Fichier de sortie

# Choix de la méthode (1, 2, 3, 4, ou "all" pour tout tester)
METHOD = 3  # Options: 1, 2, 3, 4, "all"

# ===== MÉTHODE 1: Statistical Outlier Removal (SOR) =====
# Retire les points isolés basé sur la distribution statistique
SOR_NB_NEIGHBORS = 20      # Nombre de voisins à considérer (↑ = plus strict)
SOR_STD_RATIO = 2.0        # Ratio écart-type (↓ = plus strict, retire + de points)

# ===== MÉTHODE 2: Radius Outlier Removal (ROR) =====
# Retire les points qui n'ont pas assez de voisins dans un rayon donné
ROR_NB_POINTS = 16         # Nombre minimum de voisins requis (↑ = plus strict)
ROR_RADIUS = 0.05          # Rayon de recherche en mètres (↓ = plus strict)

# ===== MÉTHODE 3: Combinaison SOR + ROR =====
# Applique d'abord SOR puis ROR pour un nettoyage progressif
COMBO_SOR_NEIGHBORS = 50
COMBO_SOR_STD = 2.5        # Plus permissif pour le premier passage
COMBO_ROR_POINTS = 10
COMBO_ROR_RADIUS = 0.3

# ===== MÉTHODE 4: DBSCAN Clustering =====
# Garde seulement les clusters principaux, vire les points isolés (noise)
DBSCAN_EPS = 0.05          # Distance max entre 2 points d'un même cluster (↓ = plus strict)
DBSCAN_MIN_POINTS = 10     # Nombre min de points pour former un cluster (↑ = plus strict)
DBSCAN_KEEP_TOP_N = 1      # Garder les N plus gros clusters

# ==================== FONCTIONS ====================

def print_stats(pcd, name="Point Cloud"):
    """Affiche les statistiques du nuage de points."""
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
    
    # Clustering DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # -1 = bruit, 0+ = clusters
    noise_count = np.sum(labels == -1)
    num_clusters = labels.max() + 1
    
    print(f"  → {num_clusters} clusters trouvés, {noise_count} points de bruit")
    
    if num_clusters == 0:
        print("  [ATTENTION] Aucun cluster trouvé ! Relâche les paramètres.")
        return pcd, np.arange(len(pcd.points))
    
    # Compter les points par cluster
    cluster_sizes = []
    for i in range(num_clusters):
        size = np.sum(labels == i)
        cluster_sizes.append((i, size))
    
    # Trier par taille décroissante
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Garder les N plus gros
    keep_n = min(keep_top_n, len(cluster_sizes))
    selected_clusters = [cluster_sizes[i][0] for i in range(keep_n)]
    
    print(f"  Clusters gardés: {selected_clusters} (tailles: {[cluster_sizes[i][1] for i in range(keep_n)]})")
    
    # Créer le masque
    mask = np.zeros(len(labels), dtype=bool)
    for cluster_id in selected_clusters:
        mask |= (labels == cluster_id)
    
    indices = np.where(mask)[0]
    pcd_clean = pcd.select_by_index(indices)
    
    removed = len(pcd.points) - len(pcd_clean.points)
    percent = (removed / len(pcd.points)) * 100
    print(f"  → Retiré: {removed} pts ({percent:.1f}%)")
    
    return pcd_clean, indices

def visualize_result(cleaned, title="Résultat"):
    """Visualise juste le résultat nettoyé."""
    print(f"[Visualisation] {title}")
    
    # Copier pour ne pas modifier l'original
    pcd_clean = o3d.geometry.PointCloud(cleaned)
    
    # Colorier en rouge
    pcd_clean.paint_uniform_color([1.0, 0.3, 0.3])
    
    o3d.visualization.draw_geometries(
        [pcd_clean],
        window_name=title,
        width=1280,
        height=720
    )

# ==================== MAIN ====================

print(f"\n[Chargement] {INPUT_FILE}")
pcd_original = o3d.io.read_point_cloud(INPUT_FILE)

if len(pcd_original.points) == 0:
    print("[ERREUR] Le fichier est vide ou n'a pas pu être chargé!")
    exit(1)

print_stats(pcd_original, "NUAGE ORIGINAL")

# ==================== TEST DES MÉTHODES ====================

if METHOD == "all":
    print("\n[TEST TOUTES LES MÉTHODES]\n")
    
    # Test méthode 1
    pcd_m1, _ = method_1_sor(pcd_original, SOR_NB_NEIGHBORS, SOR_STD_RATIO)
    print_stats(pcd_m1, "Résultat")
    
    # Test méthode 2
    print()
    pcd_m2, _ = method_2_ror(pcd_original, ROR_NB_POINTS, ROR_RADIUS)
    print_stats(pcd_m2, "Résultat")
    
    # Test méthode 3
    print()
    pcd_m3, _ = method_3_combo(
        pcd_original, 
        COMBO_SOR_NEIGHBORS, 
        COMBO_SOR_STD,
        COMBO_ROR_POINTS, 
        COMBO_ROR_RADIUS
    )
    print_stats(pcd_m3, "Résultat")
    
    # Test méthode 4
    print()
    pcd_m4, _ = method_4_dbscan(pcd_original, DBSCAN_EPS, DBSCAN_MIN_POINTS, DBSCAN_KEEP_TOP_N)
    print_stats(pcd_m4, "Résultat")
    
    # Sauvegarder toutes les versions
    o3d.io.write_point_cloud("denoised_method1_sor.ply", pcd_m1)
    o3d.io.write_point_cloud("denoised_method2_ror.ply", pcd_m2)
    o3d.io.write_point_cloud("denoised_method3_combo.ply", pcd_m3)
    o3d.io.write_point_cloud("denoised_method4_dbscan.ply", pcd_m4)
    print("\n[Sauvegardé] denoised_method1_sor.ply, denoised_method2_ror.ply, denoised_method3_combo.ply, denoised_method4_dbscan.ply")
    
    # Visualisations des résultats
    visualize_result(pcd_m1, "Méthode 1: SOR")
    visualize_result(pcd_m2, "Méthode 2: ROR")
    visualize_result(pcd_m3, "Méthode 3: COMBO")
    visualize_result(pcd_m4, "Méthode 4: DBSCAN")
    
    # Sauvegarder la meilleure (dbscan) comme fichier principal
    pcd_final = pcd_m4
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Sauvegardé] {OUTPUT_FILE}")

elif METHOD == 1:
    pcd_final, _ = method_1_sor(pcd_original, SOR_NB_NEIGHBORS, SOR_STD_RATIO)
    print_stats(pcd_final, "Résultat")
    visualize_result(pcd_final, "SOR")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Sauvegardé] {OUTPUT_FILE}")

elif METHOD == 2:
    pcd_final, _ = method_2_ror(pcd_original, ROR_NB_POINTS, ROR_RADIUS)
    print_stats(pcd_final, "Résultat")
    visualize_result(pcd_final, "ROR")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Sauvegardé] {OUTPUT_FILE}")

elif METHOD == 3:
    pcd_final, _ = method_3_combo(
        pcd_original,
        COMBO_SOR_NEIGHBORS,
        COMBO_SOR_STD,
        COMBO_ROR_POINTS,
        COMBO_ROR_RADIUS
    )
    print_stats(pcd_final, "Résultat")
    visualize_result(pcd_final, "COMBO")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Sauvegardé] {OUTPUT_FILE}")

elif METHOD == 4:
    pcd_final, _ = method_4_dbscan(pcd_original, DBSCAN_EPS, DBSCAN_MIN_POINTS, DBSCAN_KEEP_TOP_N)
    print_stats(pcd_final, "Résultat")
    visualize_result(pcd_final, "DBSCAN")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_final)
    print(f"[Sauvegardé] {OUTPUT_FILE}")

print("\n[✓] Done!\n")
