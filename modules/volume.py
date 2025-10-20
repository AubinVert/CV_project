# ==========================================================
#   Module 4: Volume Estimation
# ==========================================================

import numpy as np
import open3d as o3d
from pathlib import Path
import config

def run_volume_estimation(input_file=None, visualize=None):
    """
    Estimate volume of segmented extinguisher.
    
    Args:
        input_file: Input file path (default: config.CLEAN_CLOUD)
        visualize: Display visualization (default: config.VISUALIZE)
    
    Returns:
        dict: Estimation results {
            'convex_hull_m3': float,
            'convex_hull_L': float,
            'cylinder_m3': float,
            'cylinder_L': float,
            'in_target': bool
        }
    """
    print("\n" + "="*70)
    print("MODULE 4: VOLUME ESTIMATION")
    print("="*70)
    
    # Default values
    input_file = input_file or config.CLEAN_CLOUD
    visualize = visualize if visualize is not None else config.VISUALIZE
    
    print(f"\n[Loading] {input_file}")
    pcd_clean = o3d.io.read_point_cloud(str(input_file))
    
    if pcd_clean.is_empty():
        raise RuntimeError(f"Failed to load {input_file}")
    
    print(f"Loaded: {len(pcd_clean.points)} points")
    
    # ==========================================================
    # Volume estimation (Convex Hull + Cylinder)
    # ==========================================================
    print("\n=== Volume Estimation ===")
    pts = np.asarray(pcd_clean.points)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins
    print(f"Bounding box (X,Y,Z): {dims}")
    
    # Calculate height and diameter
    height_raw = np.max(dims)
    height_adjusted = height_raw - config.CYLINDER_HEIGHT_MARGIN_TOP - config.CYLINDER_HEIGHT_MARGIN_BOTTOM
    
    sorted_dims = np.sort(dims)
    diameter_raw = np.mean(sorted_dims[:2])
    radius_raw = diameter_raw / 2.0
    radius_adjusted = radius_raw * config.CYLINDER_RADIUS_FACTOR
    
    print(f"Height (raw): {height_raw:.3f} m")
    print(f"Height (adjusted): {height_adjusted:.3f} m")
    print(f"Radius (raw): {radius_raw:.3f} m")
    print(f"Radius (adjusted x{config.CYLINDER_RADIUS_FACTOR}): {radius_adjusted:.3f} m")
    
    # ==========================================================
    # PCA: Find principal axis for proper alignment
    # ==========================================================
    print("\n=== PCA Alignment ===")
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    
    # Compute covariance matrix and eigenvectors
    cov_matrix = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (largest = principal axis)
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx_sorted]
    principal_axis = eigenvectors[:, 0]  # Main axis (height)
    
    # Project points onto principal axis to get height
    projections = np.dot(pts_centered, principal_axis)
    height_pca = projections.max() - projections.min()
    height_adjusted = height_pca - config.CYLINDER_HEIGHT_MARGIN_TOP - config.CYLINDER_HEIGHT_MARGIN_BOTTOM
    
    # Calculate radius from points perpendicular to axis
    # Project to plane perpendicular to principal axis
    pts_perp = pts_centered - np.outer(projections, principal_axis)
    distances = np.linalg.norm(pts_perp, axis=1)
    radius_raw = np.percentile(distances, 95)  # 95th percentile for robustness
    radius_adjusted = radius_raw * config.CYLINDER_RADIUS_FACTOR
    
    print(f"Principal axis: {principal_axis}")
    print(f"Height (PCA): {height_pca:.3f} m")
    print(f"Height (adjusted): {height_adjusted:.3f} m")
    print(f"Radius (95th percentile): {radius_raw:.3f} m")
    print(f"Radius (adjusted x{config.CYLINDER_RADIUS_FACTOR}): {radius_adjusted:.3f} m")
    
    # --- METHOD 1: Convex Hull ---
    hull, _ = pcd_clean.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color([0, 1, 0])
    
    try:
        vol_hull = hull.get_volume()
    except AttributeError:
        verts, tris = np.asarray(hull.vertices), np.asarray(hull.triangles)
        vol_hull = abs(np.sum([np.dot(verts[t[0]], np.cross(verts[t[1]], verts[t[2]])) for t in tris])) / 6.0
    
    # --- METHOD 2: Adjusted Cylinder (PCA-aligned) ---
    vol_cylinder_adjusted = np.pi * (radius_adjusted ** 2) * height_adjusted
    
    # Convert to liters
    volume_liters_hull = vol_hull * 1000
    volume_liters_cylinder = vol_cylinder_adjusted * 1000
    
    print(f"\n{'='*70}")
    print(f"[RESULTS - 2 METHODS]")
    print(f"{'='*70}")
    print(f"1. Convex Hull:")
    print(f"   → {vol_hull:.4f} m³ = {volume_liters_hull:.1f} L")
    print(f"\n2. Adjusted Cylinder (PCA-aligned, RECOMMENDED):")
    print(f"   Radius: {radius_raw:.3f} m × {config.CYLINDER_RADIUS_FACTOR} = {radius_adjusted:.3f} m")
    print(f"   Height: {height_pca:.3f} m - margins = {height_adjusted:.3f} m")
    print(f"   → {vol_cylinder_adjusted:.4f} m³ = {volume_liters_cylinder:.1f} L")
    print(f"\n[TARGET] Goal: 60-65 L ± 30% → [{config.TARGET_MIN:.1f} - {config.TARGET_MAX:.1f}] L")
    
    status_hull = "✓" if config.TARGET_MIN <= volume_liters_hull <= config.TARGET_MAX else "✗"
    status_cyl = "✓" if config.TARGET_MIN <= volume_liters_cylinder <= config.TARGET_MAX else "✗"
    
    print(f"\n[STATUS] Hull: {status_hull} | Cylinder: {status_cyl}")
    print(f"{'='*70}")
    
    # ==========================================================
    # Create PCA-aligned cylinder for visualization
    # ==========================================================
    # Create cylinder along Z-axis
    cylinder_adjusted = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius_adjusted,
        height=height_adjusted,
        resolution=50,
        split=4
    )
    
    # Create rotation matrix to align Z-axis with principal axis
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis = cross product
    rotation_axis = np.cross(z_axis, principal_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm > 1e-6:  # Not parallel
        rotation_axis = rotation_axis / rotation_axis_norm
        # Rotation angle = arccos of dot product
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, principal_axis), -1.0, 1.0))
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    else:
        # Already aligned or opposite
        if np.dot(z_axis, principal_axis) < 0:
            R = -np.eye(3)  # Flip
        else:
            R = np.eye(3)
    
    cylinder_adjusted.rotate(R, center=[0, 0, 0])
    cylinder_adjusted.translate(centroid)
    cylinder_adjusted.paint_uniform_color([0, 1, 1])
    
    cylinder_adjusted_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder_adjusted)
    cylinder_adjusted_ls.paint_uniform_color([0, 1, 1])
    
    # ==========================================================
    # Visualization
    # ==========================================================
    if visualize:
        print("\n=== Visualization ===")
        print("[LEGEND]")
        print("  Red   : Segmented extinguisher")
        print("  Green : Convex Hull")
        print("  Cyan  : PCA-aligned Cylinder")
        
        pcd_clean.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd_clean, hull_ls, cylinder_adjusted_ls],
            window_name=f"Step 4/4: Volume Estimation - Hull:{volume_liters_hull:.1f}L | Cyl:{volume_liters_cylinder:.1f}L",
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT
        )
    
    print("\n[✓] Module 4 complete\n")
    
    return {
        'convex_hull_m3': vol_hull,
        'convex_hull_L': volume_liters_hull,
        'cylinder_m3': vol_cylinder_adjusted,
        'cylinder_L': volume_liters_cylinder,
        'in_target': config.TARGET_MIN <= volume_liters_cylinder <= config.TARGET_MAX
    }

if __name__ == "__main__":
    run_volume_estimation()
