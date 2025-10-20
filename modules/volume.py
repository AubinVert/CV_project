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
    
    # --- METHOD 1: Convex Hull ---
    hull, _ = pcd_clean.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color([0, 1, 0])
    
    try:
        vol_hull = hull.get_volume()
    except AttributeError:
        verts, tris = np.asarray(hull.vertices), np.asarray(hull.triangles)
        vol_hull = abs(np.sum([np.dot(verts[t[0]], np.cross(verts[t[1]], verts[t[2]])) for t in tris])) / 6.0
    
    # --- METHOD 2: Adjusted Cylinder ---
    vol_cylinder_adjusted = np.pi * (radius_adjusted ** 2) * height_adjusted
    
    volume_liters_hull = vol_hull * 1000
    volume_liters_cylinder = vol_cylinder_adjusted * 1000
    
    print(f"\n{'='*70}")
    print(f"[RESULTS - 2 METHODS]")
    print(f"{'='*70}")
    print(f"1. Convex Hull:")
    print(f"   → {vol_hull:.4f} m³ = {volume_liters_hull:.1f} L")
    print(f"\n2. Adjusted Cylinder (RECOMMENDED):")
    print(f"   Radius: {radius_raw:.3f} m × {config.CYLINDER_RADIUS_FACTOR} = {radius_adjusted:.3f} m")
    print(f"   Height: {height_raw:.3f} m - margins = {height_adjusted:.3f} m")
    print(f"   → {vol_cylinder_adjusted:.4f} m³ = {volume_liters_cylinder:.1f} L")
    print(f"\n[TARGET] Goal: 60-65 L ± 30% → [{config.TARGET_MIN:.1f} - {config.TARGET_MAX:.1f}] L")
    
    status_hull = "✓" if config.TARGET_MIN <= volume_liters_hull <= config.TARGET_MAX else "✗"
    status_cyl = "✓" if config.TARGET_MIN <= volume_liters_cylinder <= config.TARGET_MAX else "✗"
    
    print(f"\n[STATUS] Hull: {status_hull} | Adjusted Cylinder: {status_cyl}")
    print(f"{'='*70}")
    
    # ==========================================================
    # Create cylinder for visualization
    # ==========================================================
    center = (mins + maxs) / 2.0
    axis_idx = np.argmax(dims)
    
    cylinder_adjusted = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius_adjusted,
        height=height_adjusted,
        resolution=50,
        split=4
    )
    
    # Rotate according to main axis
    if axis_idx == 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
    elif axis_idx == 1:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0])
    else:
        R = np.eye(3)
    
    cylinder_adjusted.rotate(R, center=[0, 0, 0])
    cylinder_adjusted.translate(center)
    cylinder_adjusted.paint_uniform_color([0, 1, 1])
    
    cylinder_adjusted_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder_adjusted)
    cylinder_adjusted_ls.paint_uniform_color([0, 1, 1])
    
    # ==========================================================
    # Visualization
    # ==========================================================
    if visualize:
        print("\n=== Visualization ===")
        print("[LEGEND]")
        print("  🔴 Red   : Segmented extinguisher")
        print("  🟢 Green : Convex Hull")
        print("  🩵 Cyan  : Adjusted Cylinder")
        
        pcd_clean.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd_clean, hull_ls, cylinder_adjusted_ls],
            window_name=f"Step 4/4: Volume Estimation - {volume_liters_cylinder:.1f}L (Target: 60-65L)",
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
