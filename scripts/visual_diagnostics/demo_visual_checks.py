#!/usr/bin/env python3
"""
Demonstration script showing how to use the visual diagnostic tools.

This script provides example commands for running visual checks on existing
test data and explains how to interpret the results.
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ Command completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Command failed")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ Command execution failed: {e}")

def main():
    """Main demonstration function."""
    print("DiffusePipe Visual Diagnostics - Demo Script")
    print("=" * 50)
    
    # Check if test data exists
    test_cbf = Path("747/lys_nitr_10_6_0491.cbf")
    test_expt = Path("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt")
    test_refl = Path("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl")
    
    if not all(f.exists() for f in [test_cbf, test_expt, test_refl]):
        print("\n⚠️  Test data not found!")
        print("Expected files:")
        print(f"  - {test_cbf}")
        print(f"  - {test_expt}")
        print(f"  - {test_refl}")
        print("\nTo run this demo, ensure test data is available.")
        print("This demo shows the commands you would run with real data.")
    
    # Prepare environment
    env = {
        **dict(sys.environ),
        'PYTHONPATH': 'src'
    }
    
    # Demo 1: DIALS processing check
    run_command([
        'python', 'scripts/visual_diagnostics/check_dials_processing.py',
        '--raw-image', str(test_cbf),
        '--expt', str(test_expt),
        '--refl', str(test_refl),
        '--output-dir', 'demo_dials_check',
        '--verbose'
    ], "DIALS Processing Visual Check")
    
    # Demo 2: Pixel mask check
    run_command([
        'python', 'scripts/visual_diagnostics/check_pixel_masks.py',
        '--expt', str(test_expt),
        '--images', str(test_cbf),
        '--static-config', '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 40}}',
        '--output-dir', 'demo_pixel_mask_check',
        '--verbose'
    ], "Pixel Mask Visual Check")
    
    # Demo 3: Total mask check (Option A)
    run_command([
        'python', 'scripts/visual_diagnostics/check_total_mask.py',
        '--raw-image', str(test_cbf),
        '--expt', str(test_expt),
        '--refl', str(test_refl),
        '--output-dir', 'demo_total_mask_check_a',
        '--verbose'
    ], "Total Mask Visual Check (Option A - dials.generate_mask)")
    
    # Demo 4: Total mask check (Option B)
    run_command([
        'python', 'scripts/visual_diagnostics/check_total_mask.py',
        '--raw-image', str(test_cbf),
        '--expt', str(test_expt),
        '--refl', str(test_refl),
        '--use-option-b',
        '--output-dir', 'demo_total_mask_check_b',
        '--verbose'
    ], "Total Mask Visual Check (Option B - shoebox-based)")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print("\nIf test data was available, check the following output directories:")
    print("  - demo_dials_check/")
    print("  - demo_pixel_mask_check/")
    print("  - demo_total_mask_check_a/")
    print("  - demo_total_mask_check_b/")
    print("\nEach directory contains:")
    print("  - PNG plot files for visual inspection")
    print("  - TXT info files with statistics and configuration")
    print("\nTo run on your own data, modify the file paths in the commands above.")

if __name__ == "__main__":
    main()