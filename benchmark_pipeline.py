# Important note : make sur to put VISUALIZE = False in config.py before running this script
# ==========================================================
#   Benchmark Pipeline - Multiple runs analysis
# ==========================================================

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d

# Import des modules
from modules import reconstruction, denoising, segmentation, volume

# Minimum number of points required for valid reconstruction
MIN_POINTS_THRESHOLD = 4000

def run_single_iteration(iteration_num):
    """
    Run one complete pipeline iteration.
    
    Args:
        iteration_num: Iteration number for logging
        
    Returns:
        dict: Results containing volume measurements and metadata, or None if invalid
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Module 1: COLMAP Reconstruction
        print(f"▶ [{iteration_num}] Module 1: Reconstruction...")
        raw_cloud = reconstruction.run_reconstruction()
        
        # Check number of points in reconstruction
        pcd = o3d.io.read_point_cloud(str(raw_cloud))
        num_points = len(pcd.points)
        print(f"   Points reconstructed: {num_points}")
        
        if num_points < MIN_POINTS_THRESHOLD:
            elapsed = time.time() - start_time
            print(f"\n⚠️  Iteration {iteration_num} SKIPPED: Only {num_points} points (threshold: {MIN_POINTS_THRESHOLD})")
            print(f"   This iteration will not count towards the total.\n")
            return None  # Return None to indicate invalid iteration
        
        print(f"   ✓ Valid reconstruction ({num_points} >= {MIN_POINTS_THRESHOLD} points)")
        
        # Module 2: Denoising
        print(f"▶ [{iteration_num}] Module 2: Denoising...")
        denoised_cloud = denoising.run_denoising(input_file=raw_cloud)
        
        # Module 3: Segmentation
        print(f"▶ [{iteration_num}] Module 3: Segmentation...")
        clean_cloud = segmentation.run_segmentation(input_file=denoised_cloud)
        
        # Module 4: Volume Estimation
        print(f"▶ [{iteration_num}] Module 4: Volume Estimation...")
        results = volume.run_volume_estimation(input_file=clean_cloud)
        
        elapsed = time.time() - start_time
        
        # Compile results
        iteration_result = {
            'iteration': iteration_num,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': elapsed,
            'num_points': num_points,
            'convex_hull_L': results['convex_hull_L'],
            'cylinder_L': results['cylinder_L'],
            'average_L': results['average_L'],
            'in_target': results['in_target'],
            'success': True,
            'error': None
        }
        
        print(f"\n✓ Iteration {iteration_num} completed in {elapsed:.1f}s")
        print(f"  Points:      {num_points}")
        print(f"  Convex Hull: {results['convex_hull_L']:.2f} L")
        print(f"  Cylinder:    {results['cylinder_L']:.2f} L")
        print(f"  Average:     {results['average_L']:.2f} L ⭐")
        
        return iteration_result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Iteration {iteration_num} FAILED after {elapsed:.1f}s: {e}")
        
        return {
            'iteration': iteration_num,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': elapsed,
            'num_points': 0,
            'convex_hull_L': np.nan,
            'cylinder_L': np.nan,
            'average_L': np.nan,
            'in_target': False,
            'success': False,
            'error': str(e)
        }

def analyze_results(results_list):
    """
    Analyze benchmark results and generate statistics.
    
    Args:
        results_list: List of iteration results
        
    Returns:
        dict: Statistical analysis
    """
    df = pd.DataFrame(results_list)
    
    # Filter successful runs
    df_success = df[df['success'] == True]
    n_success = len(df_success)
    n_total = len(df)
    
    if n_success == 0:
        print("\n[ERROR] No successful runs to analyze!")
        return None
    
    # Statistics
    stats = {
        'total_runs': n_total,
        'successful_runs': n_success,
        'failed_runs': n_total - n_success,
        'success_rate': (n_success / n_total) * 100,
        
        # Point cloud stats
        'points_mean': df_success['num_points'].mean(),
        'points_std': df_success['num_points'].std(),
        'points_min': df_success['num_points'].min(),
        'points_max': df_success['num_points'].max(),
        
        # Convex Hull stats
        'convex_hull_mean': df_success['convex_hull_L'].mean(),
        'convex_hull_std': df_success['convex_hull_L'].std(),
        'convex_hull_min': df_success['convex_hull_L'].min(),
        'convex_hull_max': df_success['convex_hull_L'].max(),
        'convex_hull_median': df_success['convex_hull_L'].median(),
        
        # Cylinder stats
        'cylinder_mean': df_success['cylinder_L'].mean(),
        'cylinder_std': df_success['cylinder_L'].std(),
        'cylinder_min': df_success['cylinder_L'].min(),
        'cylinder_max': df_success['cylinder_L'].max(),
        'cylinder_median': df_success['cylinder_L'].median(),
        
        # Average volume stats (RECOMMENDED)
        'average_mean': df_success['average_L'].mean(),
        'average_std': df_success['average_L'].std(),
        'average_min': df_success['average_L'].min(),
        'average_max': df_success['average_L'].max(),
        'average_median': df_success['average_L'].median(),
        
        # Duration stats
        'duration_mean': df_success['duration_seconds'].mean(),
        'duration_std': df_success['duration_seconds'].std(),
        
        # Target range
        'in_target_count': df_success['in_target'].sum(),
        'in_target_rate': (df_success['in_target'].sum() / n_success) * 100,
    }
    
    return stats, df, df_success

def print_summary(stats):
    """Print formatted summary of benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n📊 RUN STATISTICS")
    print(f"   Total runs:       {stats['total_runs']}")
    print(f"   Successful:       {stats['successful_runs']} ({stats['success_rate']:.1f}%)")
    print(f"   Failed:           {stats['failed_runs']}")
    
    print(f"\n📍 RECONSTRUCTION POINTS")
    print(f"   Mean points:      {stats['points_mean']:.0f} ± {stats['points_std']:.0f}")
    print(f"   Range:            [{stats['points_min']:.0f}, {stats['points_max']:.0f}]")
    
    print(f"\n⏱️  EXECUTION TIME")
    print(f"   Mean duration:    {stats['duration_mean']:.1f} ± {stats['duration_std']:.1f} seconds")
    
    print(f"\n📦 CONVEX HULL VOLUME")
    print(f"   Mean:             {stats['convex_hull_mean']:.2f} ± {stats['convex_hull_std']:.2f} L")
    print(f"   Median:           {stats['convex_hull_median']:.2f} L")
    print(f"   Range:            [{stats['convex_hull_min']:.2f}, {stats['convex_hull_max']:.2f}] L")
    print(f"   Variability:      {stats['convex_hull_std'] / stats['convex_hull_mean'] * 100:.1f}% CoV")
    
    print(f"\n🛢️  CYLINDER VOLUME")
    print(f"   Mean:             {stats['cylinder_mean']:.2f} ± {stats['cylinder_std']:.2f} L")
    print(f"   Median:           {stats['cylinder_median']:.2f} L")
    print(f"   Range:            [{stats['cylinder_min']:.2f}, {stats['cylinder_max']:.2f}] L")
    print(f"   Variability:      {stats['cylinder_std'] / stats['cylinder_mean'] * 100:.1f}% CoV")
    
    print(f"\n⭐ AVERAGE VOLUME (RECOMMENDED)")
    print(f"   Mean:             {stats['average_mean']:.2f} ± {stats['average_std']:.2f} L")
    print(f"   Median:           {stats['average_median']:.2f} L")
    print(f"   Range:            [{stats['average_min']:.2f}, {stats['average_max']:.2f}] L")
    print(f"   Variability:      {stats['average_std'] / stats['average_mean'] * 100:.1f}% CoV")
    
    print(f"\n🎯 TARGET RANGE (42.0 - 84.5 L)")
    print(f"   Runs in target:   {stats['in_target_count']}/{stats['successful_runs']} ({stats['in_target_rate']:.1f}%)")
    
    print("="*70 + "\n")

def plot_results(df_success, output_dir):
    """
    Generate visualization plots of benchmark results.
    
    Args:
        df_success: DataFrame with successful runs
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Box plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convex Hull
    ax1.boxplot([df_success['convex_hull_L']], tick_labels=['Convex Hull'])
    ax1.axhspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax1.set_ylabel('Volume (L)')
    ax1.set_title('Convex Hull Volume Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cylinder
    ax2.boxplot([df_success['cylinder_L']], tick_labels=['Cylinder'])
    ax2.axhspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax2.set_ylabel('Volume (L)')
    ax2.set_title('Cylinder Volume Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average (Recommended)
    ax3.boxplot([df_success['average_L']], tick_labels=['Average'])
    ax3.axhspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax3.set_ylabel('Volume (L)')
    ax3.set_title('Average Volume Distribution ⭐')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_distribution.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'volume_distribution.png'}")
    plt.close()
    
    # 2. Time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_success['iteration'], df_success['convex_hull_L'], 
            marker='o', label='Convex Hull', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(df_success['iteration'], df_success['cylinder_L'], 
            marker='s', label='Cylinder', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(df_success['iteration'], df_success['average_L'], 
            marker='D', label='Average (Recommended) ⭐', linewidth=3, markersize=10, color='purple')
    ax.axhspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Volume (L)')
    ax.set_title('Volume Measurements Across Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_time_series.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'volume_time_series.png'}")
    plt.close()
    
    # 3. Histogram with KDE
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convex Hull histogram
    ax1.hist(df_success['convex_hull_L'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(df_success['convex_hull_L'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df_success["convex_hull_L"].mean():.2f} L')
    ax1.axvspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax1.set_xlabel('Volume (L)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Convex Hull Volume Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cylinder histogram
    ax2.hist(df_success['cylinder_L'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(df_success['cylinder_L'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df_success["cylinder_L"].mean():.2f} L')
    ax2.axvspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax2.set_xlabel('Volume (L)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Cylinder Volume Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average histogram (Recommended)
    ax3.hist(df_success['average_L'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(df_success['average_L'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df_success["average_L"].mean():.2f} L')
    ax3.axvspan(42.0, 84.5, alpha=0.2, color='green', label='Target Range')
    ax3.set_xlabel('Volume (L)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Average Volume Histogram ⭐')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_histogram.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'volume_histogram.png'}")
    plt.close()
    
    # 4. Scatter plot: Convex Hull vs Cylinder
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df_success['convex_hull_L'], df_success['cylinder_L'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add diagonal line (perfect correlation)
    min_val = min(df_success['convex_hull_L'].min(), df_success['cylinder_L'].min())
    max_val = max(df_success['convex_hull_L'].max(), df_success['cylinder_L'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    ax.set_xlabel('Convex Hull Volume (L)')
    ax.set_ylabel('Cylinder Volume (L)')
    ax.set_title('Convex Hull vs Cylinder Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_correlation.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'volume_correlation.png'}")
    plt.close()

def save_results(results_list, stats, output_dir):
    """
    Save benchmark results to CSV and text files.
    
    Args:
        results_list: List of iteration results
        stats: Statistical analysis
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results to CSV
    df = pd.DataFrame(results_list)
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save summary statistics to text file
    txt_path = output_dir / 'benchmark_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BENCHMARK SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"RUN STATISTICS\n")
        f.write(f"  Total runs:       {stats['total_runs']}\n")
        f.write(f"  Successful:       {stats['successful_runs']} ({stats['success_rate']:.1f}%)\n")
        f.write(f"  Failed:           {stats['failed_runs']}\n\n")
        
        f.write(f"RECONSTRUCTION POINTS\n")
        f.write(f"  Mean points:      {stats['points_mean']:.0f} ± {stats['points_std']:.0f}\n")
        f.write(f"  Range:            [{stats['points_min']:.0f}, {stats['points_max']:.0f}]\n\n")
        
        f.write(f"EXECUTION TIME\n")
        f.write(f"  Mean duration:    {stats['duration_mean']:.1f} ± {stats['duration_std']:.1f} seconds\n\n")
        
        f.write(f"CONVEX HULL VOLUME\n")
        f.write(f"  Mean:             {stats['convex_hull_mean']:.2f} ± {stats['convex_hull_std']:.2f} L\n")
        f.write(f"  Median:           {stats['convex_hull_median']:.2f} L\n")
        f.write(f"  Range:            [{stats['convex_hull_min']:.2f}, {stats['convex_hull_max']:.2f}] L\n")
        f.write(f"  Variability:      {stats['convex_hull_std'] / stats['convex_hull_mean'] * 100:.1f}% CoV\n\n")
        
        f.write(f"CYLINDER VOLUME\n")
        f.write(f"  Mean:             {stats['cylinder_mean']:.2f} ± {stats['cylinder_std']:.2f} L\n")
        f.write(f"  Median:           {stats['cylinder_median']:.2f} L\n")
        f.write(f"  Range:            [{stats['cylinder_min']:.2f}, {stats['cylinder_max']:.2f}] L\n")
        f.write(f"  Variability:      {stats['cylinder_std'] / stats['cylinder_mean'] * 100:.1f}% CoV\n\n")
        
        f.write(f"AVERAGE VOLUME (RECOMMENDED)\n")
        f.write(f"  Mean:             {stats['average_mean']:.2f} ± {stats['average_std']:.2f} L\n")
        f.write(f"  Median:           {stats['average_median']:.2f} L\n")
        f.write(f"  Range:            [{stats['average_min']:.2f}, {stats['average_max']:.2f}] L\n")
        f.write(f"  Variability:      {stats['average_std'] / stats['average_mean'] * 100:.1f}% CoV\n\n")
        
        f.write(f"TARGET RANGE (42.0 - 84.5 L)\n")
        f.write(f"  Runs in target:   {stats['in_target_count']}/{stats['successful_runs']} ({stats['in_target_rate']:.1f}%)\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Saved: {txt_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline: run multiple iterations and analyze variability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 iterations (default)
  python benchmark_pipeline.py
  
  # Run 20 iterations
  python benchmark_pipeline.py -n 20
  
  # Custom output directory
  python benchmark_pipeline.py -n 15 -o results/benchmark_2025
        """
    )
    
    parser.add_argument(
        '-n', '--iterations',
        type=int,
        default=10,
        help='Number of iterations to run (default: 10)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    if args.iterations < 1:
        print("[ERROR] Number of iterations must be at least 1")
        return 1
    
    print("\n" + "="*70)
    print("BENCHMARK PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target iterations:     {args.iterations}")
    print(f"  Min points threshold:  {MIN_POINTS_THRESHOLD}")
    print(f"  Output dir:            {args.output}")
    print(f"  Plots:                 {'No' if args.no_plots else 'Yes'}")
    print("="*70 + "\n")
    
    # Run iterations (keep going until we have enough valid ones)
    results_list = []
    valid_count = 0
    attempt_num = 0
    max_attempts = args.iterations * 5  # Safety limit to avoid infinite loop
    benchmark_start = time.time()
    
    while valid_count < args.iterations and attempt_num < max_attempts:
        attempt_num += 1
        result = run_single_iteration(attempt_num)
        
        if result is not None:  # Valid iteration
            valid_count += 1
            # Renumber iteration to sequential valid count
            result['iteration'] = valid_count
            results_list.append(result)
            print(f"   → Valid iterations: {valid_count}/{args.iterations}")
        # If None, it was skipped due to low point count
    
    total_time = time.time() - benchmark_start
    
    if valid_count < args.iterations:
        print(f"\n[WARNING] Only got {valid_count} valid iterations after {attempt_num} attempts")
        print(f"          (Max attempts reached: {max_attempts})")
    
    # Analyze results
    print("\n" + "="*70)
    print("ANALYZING RESULTS...")
    print("="*70)
    
    analysis = analyze_results(results_list)
    if analysis is None:
        return 1
    
    stats, df, df_success = analysis
    
    # Print summary
    print_summary(stats)
    print(f"⏱️  Total benchmark time: {total_time / 60:.1f} minutes")
    print(f"   Total attempts: {attempt_num} (skipped: {attempt_num - valid_count})")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS...")
    print("="*70 + "\n")
    save_results(results_list, stats, args.output)
    
    # Generate plots
    if not args.no_plots and len(df_success) > 0:
        print("\n" + "="*70)
        print("GENERATING PLOTS...")
        print("="*70 + "\n")
        try:
            plot_results(df_success, args.output)
        except Exception as e:
            print(f"[WARNING] Failed to generate plots: {e}")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETED")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
