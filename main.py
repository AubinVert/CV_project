#!/usr/bin/env python3
# ==========================================================
#   Main Pipeline - Reconstruction & Segmentation Extincteur
# ==========================================================

import argparse
import sys
from pathlib import Path

# Import des modules
from modules import reconstruction, denoising, segmentation, volume

def run_full_pipeline():
    """Run complete pipeline from A to Z."""
    print("\n" + "="*70)
    print("FULL PIPELINE - EXTINGUISHER RECONSTRUCTION & SEGMENTATION")
    print("="*70 + "\n")
    
    try:
        # Module 1: COLMAP Reconstruction
        print("▶ Starting Module 1...")
        raw_cloud = reconstruction.run_reconstruction()
        
        # Module 2: Denoising
        print("▶ Starting Module 2...")
        denoised_cloud = denoising.run_denoising(input_file=raw_cloud)
        
        # Module 3: Segmentation
        print("▶ Starting Module 3...")
        clean_cloud = segmentation.run_segmentation(input_file=denoised_cloud)
        
        # Module 4: Volume Estimation
        print("▶ Starting Module 4...")
        results = volume.run_volume_estimation(input_file=clean_cloud)
        
        # Final summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Estimated volume (cylinder): {results['cylinder_L']:.1f} L")
        print(f"Within target (42-84.5 L): {'✓ YES' if results['in_target'] else '✗ NO'}")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline interrupted: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Extinguisher reconstruction and segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python main.py
  
  # Individual modules
  python main.py --module reconstruction
  python main.py --module denoising
  python main.py --module segmentation
  python main.py --module volume
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=['reconstruction', 'denoising', 'segmentation', 'volume'],
        help='Run a specific module only'
    )
    
    args = parser.parse_args()
    
    # Execute according to choice
    if args.module == 'reconstruction':
        reconstruction.run_reconstruction()
    elif args.module == 'denoising':
        denoising.run_denoising()
    elif args.module == 'segmentation':
        segmentation.run_segmentation()
    elif args.module == 'volume':
        volume.run_volume_estimation()
    else:
        # Full pipeline by default
        return run_full_pipeline()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
