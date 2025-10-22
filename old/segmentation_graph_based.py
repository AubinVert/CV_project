# ==========================================================
#   Graph-Based Segmentation for Sparse Point Clouds
# ==========================================================
# Method adapted for SPARSE point clouds
# Creates a spatial graph and extracts connected components
# ==========================================================

# https://www.youtube.com/watch?v=r87ToGLDoIg

import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial import KDTree

# ==================== PARAMETERS ====================
INPUT_FILE = "../dense/fused_colmap.ply"
# INPUT_FILE = "dense/fused_colmap.ply"

# Graph construction parameters
GRAPH_RADIUS = 0.5          # Connection radius between points (3 cm)
GRAPH_MAX_NEIGHBORS = 15     # Max neighbors per point
PRUNE_TO_K_NEIGHBORS = 10    # Keep only K nearest neighbors

# Component filtering
MIN_COMPONENT_SIZE = 50      # Minimum size for valid component
KEEP_LARGEST_N = 1           # Keep N largest components

# Color filtering (optional)
USE_COLOR_PREFILTER = False
RED_H_LOW_MAX = 0.05
RED_H_HIGH_MIN = 0.96
RED_S_MIN = 0.10
RED_V_MIN = 0.10

# ==================== HELPER FUNCTIONS ====================

def rgb_to_hsv(rgb):
    """Convert RGB to HSV."""
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
    """Mask for red points."""
    h, s, v = colors_hsv[:, 0], colors_hsv[:, 1], colors_hsv[:, 2]
    hue_mask = ((h <= RED_H_LOW_MAX) | (h >= RED_H_HIGH_MIN))
    sat_mask = s >= RED_S_MIN
    val_mask = v >= RED_V_MIN
    return hue_mask & sat_mask & val_mask

def build_radius_graph(points, radius, max_neighbors=None):
    """
    Build a graph by connecting points within a given radius.
    
    Args:
        points: Numpy array (N, 3) of 3D coordinates
        radius: Search radius
        max_neighbors: Max neighbors (None = unlimited)
    
    Returns:
        networkx.Graph with positions and distances
    """
    print(f"  Building graph (radius={radius:.3f}m, max_neighbors={max_neighbors})...")

    points = np.array(points)

    # Create KDTree for efficient neighbor search
    kdtree = KDTree(points)
    
    # Initialize graph
    graph = nx.Graph()
    
    # Add nodes with their positions
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    
    # Query kd tree for all points within radius
    pairs = kdtree.query_pairs(radius)

    # Add edges to graph with distances as weights
    for i, j in pairs:
        distance = np.linalg.norm(points[i] - points[j])
        graph.add_edge(i, j, weight=distance)

    # If max_neighbors is specified, prune the graph
    if max_neighbors is not None:
        graph = prune_to_k_neighbors(graph, max_neighbors)
    
    return graph

def prune_to_k_neighbors(G, k):
    """
    Reduce graph to keep only K nearest neighbors.
    
    Args:
        G: networkx.Graph
        k: Number of neighbors to keep
    
    Returns:
        Pruned networkx.Graph
    """
    print(f"    Internal pruning (keep {k} nearest neighbors)...")
    
    edges_to_remove = []
    
    for node in G.nodes():
        # Get all edges of node with their weights
        edges = [(node, neighbor, G[node][neighbor]['weight']) 
                 for neighbor in G[node]]
        
        # If more than k neighbors, keep only k nearest
        if len(edges) > k:
            # Sort by distance (weight)
            edges.sort(key=lambda x: x[2])
            # Mark edges to remove (beyond k nearest)
            edges_to_remove.extend([(e[0], e[1]) for e in edges[k:]])
    
    # Remove edges
    G.remove_edges_from(edges_to_remove)
    print(f"      → {len(edges_to_remove)} edges removed")
    
    return G

def analyze_components(G):
    """
    Analyze connected component properties.
    
    Args:
        G: networkx.Graph
    
    Returns:
        dict with statistics
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
    Visualize graph components in 3D with Open3D (much faster!).
    Display points colored by component + graph edges.
    
    Args:
        G: networkx.Graph
        points: Numpy array (N, 3) of coordinates
        title: Visualization title
    """
    print(f"  Preparing Open3D visualization...")
    
    components = list(nx.connected_components(G))
    
    # Generate random colors for each component
    np.random.seed(42)
    component_colors = np.random.rand(len(components), 3)
    
    # Create color array for each point
    point_colors = np.zeros((len(points), 3))
    for comp_idx, component in enumerate(components):
        for node_idx in component:
            point_colors[node_idx] = component_colors[comp_idx]
    
    # Create point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Create lines for graph edges
    lines = []
    line_colors = []
    
    for edge in G.edges():
        node1, node2 = edge
        lines.append([node1, node2])
        # Color = average of two node colors
        edge_color = (point_colors[node1] + point_colors[node2]) / 2.0
        line_colors.append(edge_color)
    
    # Create LineSet for edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    print(f"    → {len(components)} components, {len(lines)} edges")
    print(f"  Displaying: {title}")
    
    # Visualization
    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name=title,
        width=1280,
        height=720,
        point_show_normal=False
    )

def calculate_volume(pcd):
    """Calculate approximate volume (cylinder)."""
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
    return volume * 1000  # in liters

# ==================== MAIN PIPELINE ====================

print("\n" + "="*70)
print("SPATIAL GRAPH SEGMENTATION (Sparse Point Clouds)")
print("="*70 + "\n")

# 1. Loading
print("Step 1: Loading point cloud...")
pcd = o3d.io.read_point_cloud(INPUT_FILE)
print(f"  Loaded: {len(pcd.points)} points")

# 2. Color pre-filtering (optional)
if USE_COLOR_PREFILTER and pcd.has_colors():
    print("\nStep 2: Pre-filtering by red color...")
    colors = np.asarray(pcd.colors).clip(0, 1)
    hsv = rgb_to_hsv(colors)
    red_mask = keep_red_points(hsv)
    pcd = pcd.select_by_index(np.where(red_mask)[0])
    print(f"  After filtering: {len(pcd.points)} red points")

# Convert to numpy array
points = np.asarray(pcd.points)
print(f"\nNumber of points to process: {len(points)}")

# 3. Graph construction
print("\nStep 3: Building spatial graph...")
G = build_radius_graph(points, radius=GRAPH_RADIUS, max_neighbors=GRAPH_MAX_NEIGHBORS)

print(f"  → Graph created: {len(G.nodes)} nodes, {len(G.edges)} edges")

# 4. Connected components analysis (no separate pruning as already done)
print("\nStep 4: Extracting connected components...")
analysis = analyze_components(G)

print(f"\n{'='*70}")
print("COMPONENT ANALYSIS")
print(f"{'='*70}")
print(f"  Number of components: {analysis['num_components']}")
print(f"  Largest component: {analysis['largest_component_size']} points")
print(f"  Smallest component: {analysis['smallest_component_size']} points")
print(f"  Average size: {analysis['avg_component_size']:.1f} points")
print(f"  Isolated points: {analysis['isolated_points']}")
print(f"{'='*70}\n")

# Display component sizes
components_sorted = sorted(analysis['components'], key=len, reverse=True)
print("Top 10 largest components:")
for i, comp in enumerate(components_sorted[:10]):
    print(f"  #{i+1}: {len(comp)} points")

# 6. Filtering and selection
print(f"\nStep 5: Filtering components (min_size={MIN_COMPONENT_SIZE})...")
filtered_components = [c for c in components_sorted if len(c) >= MIN_COMPONENT_SIZE]
print(f"  {len(filtered_components)} valid components found")

# Keep N largest
if len(filtered_components) > KEEP_LARGEST_N:
    selected_components = filtered_components[:KEEP_LARGEST_N]
else:
    selected_components = filtered_components

print(f"  → Keeping the {len(selected_components)} largest component(s)")

# 7. Reconstruction of segmented point cloud
print("\nStep 6: Reconstructing point cloud...")
results = []

for i, component in enumerate(selected_components):
    component_idx = list(component)
    pcd_component = pcd.select_by_index(component_idx)
    
    # Color each component differently
    if i == 0:
        pcd_component.paint_uniform_color([1, 0, 0])  # Red for main
    elif i == 1:
        pcd_component.paint_uniform_color([0, 1, 0])  # Green
    else:
        pcd_component.paint_uniform_color([0, 0, 1])  # Blue
    
    volume = calculate_volume(pcd_component)
    results.append((f"Component {i+1}", pcd_component, len(component), volume))
    
    print(f"  Component {i+1}: {len(component)} points, Volume: {volume:.1f} L")

# 8. Graph visualization with Open3D
print("\nStep 7: Graph visualization (Open3D)...")
plot_graph_components_open3d(G, points, title=f"Graph with {analysis['num_components']} components")

# 9. Segmented point cloud visualization
print("\nStep 8: Segmented point cloud visualization...")
geometries_to_show = [res[1] for res in results]

if len(geometries_to_show) > 0:
    o3d.visualization.draw_geometries(
        geometries_to_show,
        window_name="Graph-Based Segmentation Result",
        width=1280,
        height=720
    )

# 10. Final results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
target_min, target_max = 42.0, 84.5  # 60-65 L ± 30%

for name, pcd_comp, num_points, volume in results:
    status = "✓ TARGET" if target_min <= volume <= target_max else "✗ Out of range"
    print(f"{name:20s} | {num_points:6d} pts | {volume:6.1f} L | {status}")

print(f"{'='*70}")
print(f"Target: 60-65 L ± 30% → [{target_min:.1f} - {target_max:.1f}] L")

# 11. Save
if results:
    main_component = results[0][1]
    o3d.io.write_point_cloud("extinguisher_graph_segmented.ply", main_component)
    print("\n[OK] Saved: extinguisher_graph_segmented.ply")

print("\n[✓] Pipeline complete.\n")
