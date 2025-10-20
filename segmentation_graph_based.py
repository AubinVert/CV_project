# ==========================================================
#   Graph-Based Segmentation for Sparse Point Clouds
# ==========================================================
# Méthode adaptée aux nuages de points SPARSE
# Crée un graphe spatial et extrait les composantes connexes
# ==========================================================

# https://www.youtube.com/watch?v=r87ToGLDoIg

import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial import KDTree

# ==================== PARAMÈTRES ====================
INPUT_FILE = "sparse/extinguisher_denoised.ply"

# Graph construction parameters
GRAPH_RADIUS = 0.5          # Rayon de connexion entre points (3 cm)
GRAPH_MAX_NEIGHBORS = 15     # Nombre max de voisins par point
PRUNE_TO_K_NEIGHBORS = 10    # Garder seulement les K plus proches voisins

# Filtrage des composantes
MIN_COMPONENT_SIZE = 50      # Taille minimale d'une composante valide
KEEP_LARGEST_N = 1           # Garder les N plus grandes composantes

# Filtrage couleur (optionnel)
USE_COLOR_PREFILTER = False
RED_H_LOW_MAX = 0.05
RED_H_HIGH_MIN = 0.96
RED_S_MIN = 0.10
RED_V_MIN = 0.10

# ==================== HELPER FUNCTIONS ====================

def rgb_to_hsv(rgb):
    """Conversion RGB vers HSV."""
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
    """Masque pour les points rouges."""
    h, s, v = colors_hsv[:, 0], colors_hsv[:, 1], colors_hsv[:, 2]
    hue_mask = ((h <= RED_H_LOW_MAX) | (h >= RED_H_HIGH_MIN))
    sat_mask = s >= RED_S_MIN
    val_mask = v >= RED_V_MIN
    return hue_mask & sat_mask & val_mask

def build_radius_graph(points, radius, max_neighbors=None):
    """
    Construit un graphe en connectant les points dans un rayon donné.
    
    Args:
        points: Array numpy (N, 3) des coordonnées 3D
        radius: Rayon de recherche
        max_neighbors: Nombre max de voisins (None = illimité)
    
    Returns:
        networkx.Graph avec positions et distances
    """
    print(f"  Construction du graphe (radius={radius:.3f}m, max_neighbors={max_neighbors})...")

    points = np.array(points)

    # Créer le KDTree pour recherche efficace des voisins
    kdtree = KDTree(points)
    
    # Initialiser le graphe
    graph = nx.Graph()
    
    # Ajouter les nœuds avec leurs positions
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    
    # query the kd tree for all points within radius
    pairs = kdtree.query_pairs(radius)

    # add edges to the graph with distances as weights
    for i, j in pairs:
        distance = np.linalg.norm(points[i] - points[j])
        graph.add_edge(i, j, weight=distance)

    # if max_neighbors is specified, prune the graph
    if max_neighbors is not None:
        graph = prune_to_k_neighbors(graph, max_neighbors)
    
    return graph

def prune_to_k_neighbors(G, k):
    """
    Réduit le graphe pour garder seulement les K plus proches voisins.
    
    Args:
        G: networkx.Graph
        k: Nombre de voisins à garder
    
    Returns:
        networkx.Graph pruné
    """
    print(f"    Pruning interne (garder {k} plus proches voisins)...")
    
    edges_to_remove = []
    
    for node in G.nodes():
        # Récupérer toutes les arêtes du nœud avec leurs poids
        edges = [(node, neighbor, G[node][neighbor]['weight']) 
                 for neighbor in G[node]]
        
        # Si plus de k voisins, garder seulement les k plus proches
        if len(edges) > k:
            # Trier par distance (poids)
            edges.sort(key=lambda x: x[2])
            # Marquer les arêtes à supprimer (au-delà des k plus proches)
            edges_to_remove.extend([(e[0], e[1]) for e in edges[k:]])
    
    # Supprimer les arêtes
    G.remove_edges_from(edges_to_remove)
    print(f"      → {len(edges_to_remove)} arêtes supprimées")
    
    return G

def analyze_components(G):
    """
    Analyse les propriétés des composantes connexes.
    
    Args:
        G: networkx.Graph
    
    Returns:
        dict avec les statistiques
    """
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    analysis = {
        'num_components': len(components),
        'component_sizes': component_sizes,
        'largest_component_size': max(component_sizes) if component_sizes else 0,
        'smallest_component_size': min(component_sizes) if component_sizes else 0,
        'avg_component_size': np.mean(component_sizes) if component_sizes else 0,
        'isolated_points': sum(1 for s in component_sizes if s == 1),
        'components': components
    }
    
    return analysis

def plot_graph_components_open3d(G, points, title="Graph Components"):
    """
    Visualise les composantes du graphe en 3D avec Open3D (beaucoup plus rapide !).
    Affiche les points colorés par composante + les arêtes du graphe.
    
    Args:
        G: networkx.Graph
        points: Array numpy (N, 3) des coordonnées
        title: Titre de la visualisation
    """
    print(f"  Préparation de la visualisation Open3D...")
    
    components = list(nx.connected_components(G))
    
    # Générer des couleurs aléatoires pour chaque composante
    np.random.seed(42)
    component_colors = np.random.rand(len(components), 3)
    
    # Créer un array de couleurs pour chaque point
    point_colors = np.zeros((len(points), 3))
    for comp_idx, component in enumerate(components):
        for node_idx in component:
            point_colors[node_idx] = component_colors[comp_idx]
    
    # Créer le nuage de points avec couleurs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Créer les lignes pour les arêtes du graphe
    lines = []
    line_colors = []
    
    for edge in G.edges():
        node1, node2 = edge
        lines.append([node1, node2])
        # Couleur = moyenne des couleurs des deux nœuds
        edge_color = (point_colors[node1] + point_colors[node2]) / 2.0
        line_colors.append(edge_color)
    
    # Créer le LineSet pour les arêtes
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    print(f"    → {len(components)} composantes, {len(lines)} arêtes")
    print(f"  Affichage: {title}")
    
    # Visualisation
    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name=title,
        width=1280,
        height=720,
        point_show_normal=False
    )

def calculate_volume(pcd):
    """Calcule le volume approximatif (cylindre)."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 0.0
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins
    height = np.max(dims)
    sorted_dims = np.sort(dims)
    diameter = np.mean(sorted_dims[:2])
    radius = diameter / 2.0
    volume = np.pi * (radius ** 2) * height
    return volume * 1000  # en litres

# ==================== MAIN PIPELINE ====================

print("\n" + "="*70)
print("SEGMENTATION PAR GRAPHE SPATIAL (Sparse Point Clouds)")
print("="*70 + "\n")

# 1. Chargement
print("Step 1: Chargement du nuage de points...")
pcd = o3d.io.read_point_cloud(INPUT_FILE)
print(f"  Chargé: {len(pcd.points)} points")

# 2. Préfiltrage couleur (optionnel)
if USE_COLOR_PREFILTER and pcd.has_colors():
    print("\nStep 2: Préfiltrage par couleur rouge...")
    colors = np.asarray(pcd.colors).clip(0, 1)
    hsv = rgb_to_hsv(colors)
    red_mask = keep_red_points(hsv)
    pcd = pcd.select_by_index(np.where(red_mask)[0])
    print(f"  Après filtrage: {len(pcd.points)} points rouges")

# Convertir en numpy array
points = np.asarray(pcd.points)
print(f"\nNombre de points à traiter: {len(points)}")

# 3. Construction du graphe
print("\nStep 3: Construction du graphe spatial...")
G = build_radius_graph(points, radius=GRAPH_RADIUS, max_neighbors=GRAPH_MAX_NEIGHBORS)

print(f"  → Graphe créé: {len(G.nodes)} nœuds, {len(G.edges)} arêtes")

# 4. Analyse des composantes connexes (pas de pruning séparé car déjà fait)
print("\nStep 4: Extraction des composantes connexes...")
analysis = analyze_components(G)

print(f"\n{'='*70}")
print("ANALYSE DES COMPOSANTES")
print(f"{'='*70}")
print(f"  Nombre de composantes: {analysis['num_components']}")
print(f"  Plus grande composante: {analysis['largest_component_size']} points")
print(f"  Plus petite composante: {analysis['smallest_component_size']} points")
print(f"  Taille moyenne: {analysis['avg_component_size']:.1f} points")
print(f"  Points isolés: {analysis['isolated_points']}")
print(f"{'='*70}\n")

# Afficher les tailles des composantes
components_sorted = sorted(analysis['components'], key=len, reverse=True)
print("Top 10 des plus grandes composantes:")
for i, comp in enumerate(components_sorted[:10]):
    print(f"  #{i+1}: {len(comp)} points")

# 6. Filtrage et sélection
print(f"\nStep 5: Filtrage des composantes (min_size={MIN_COMPONENT_SIZE})...")
filtered_components = [c for c in components_sorted if len(c) >= MIN_COMPONENT_SIZE]
print(f"  {len(filtered_components)} composantes valides trouvées")

# Garder les N plus grandes
if len(filtered_components) > KEEP_LARGEST_N:
    selected_components = filtered_components[:KEEP_LARGEST_N]
else:
    selected_components = filtered_components

print(f"  → Garder les {len(selected_components)} plus grande(s) composante(s)")

# 7. Reconstruction du nuage de points segmenté
print("\nStep 6: Reconstruction du nuage de points...")
results = []

for i, component in enumerate(selected_components):
    component_idx = list(component)
    pcd_component = pcd.select_by_index(component_idx)
    
    # Colorier différemment chaque composante
    if i == 0:
        pcd_component.paint_uniform_color([1, 0, 0])  # Rouge pour la principale
    elif i == 1:
        pcd_component.paint_uniform_color([0, 1, 0])  # Vert
    else:
        pcd_component.paint_uniform_color([0, 0, 1])  # Bleu
    
    volume = calculate_volume(pcd_component)
    results.append((f"Composante {i+1}", pcd_component, len(component), volume))
    
    print(f"  Composante {i+1}: {len(component)} points, Volume: {volume:.1f} L")

# 8. Visualisation du graphe avec Open3D
print("\nStep 7: Visualisation du graphe (Open3D)...")
plot_graph_components_open3d(G, points, title=f"Graph avec {analysis['num_components']} composantes")

# 9. Visualisation du nuage de points segmenté
print("\nStep 8: Visualisation du nuage de points segmenté...")
geometries_to_show = [res[1] for res in results]

if len(geometries_to_show) > 0:
    o3d.visualization.draw_geometries(
        geometries_to_show,
        window_name="Graph-Based Segmentation Result",
        width=1280,
        height=720
    )

# 10. Résultats finaux
print(f"\n{'='*70}")
print("RÉSULTATS FINAUX")
print(f"{'='*70}")
target_min, target_max = 42.0, 84.5  # 60-65 L ± 30%

for name, pcd_comp, num_points, volume in results:
    status = "✓ CIBLE" if target_min <= volume <= target_max else "✗ Hors cible"
    print(f"{name:20s} | {num_points:6d} pts | {volume:6.1f} L | {status}")

print(f"{'='*70}")
print(f"Objectif: 60-65 L ± 30% → [{target_min:.1f} - {target_max:.1f}] L")

# 11. Sauvegarde
if results:
    main_component = results[0][1]
    o3d.io.write_point_cloud("extinguisher_graph_segmented.ply", main_component)
    print("\n[OK] Sauvegardé: extinguisher_graph_segmented.ply")

print("\n[✓] Pipeline terminé.\n")
