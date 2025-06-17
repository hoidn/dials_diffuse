#!/usr/bin/env python3
"""
Check crystal orientations between two experiment files to verify reference-based indexing.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_orientations():
    """Check crystal orientations from the two experiment files."""
    
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        
        # Load the single sequence experiment file (both images use the same one)
        expt1_path = "test_sequence_e2e_output/phase3_e2e_outputs/sequence_processing/indexed_refined_detector.expt"
        expt2_path = "test_sequence_e2e_output/phase3_e2e_outputs/sequence_processing/indexed_refined_detector.expt"
        
        experiments1 = ExperimentListFactory.from_json_file(expt1_path)
        experiments2 = ExperimentListFactory.from_json_file(expt2_path)
        
        crystal1 = experiments1[0].crystal
        crystal2 = experiments2[0].crystal
        
        print("=== Crystal Orientation Analysis ===")
        print(f"Experiment 1: {expt1_path}")
        print(f"Experiment 2: {expt2_path}")
        print()
        
        # Get unit cell parameters
        uc1 = crystal1.get_unit_cell()
        uc2 = crystal2.get_unit_cell()
        
        print("Unit Cell Parameters:")
        print(f"Crystal 1: {uc1.parameters()}")
        print(f"Crystal 2: {uc2.parameters()}")
        print()
        
        # Get orientation matrices (A matrices)
        A1 = np.array(crystal1.get_A()).reshape(3, 3)
        A2 = np.array(crystal2.get_A()).reshape(3, 3)
        
        print("A Matrices:")
        print(f"Crystal 1 A matrix:\n{A1}")
        print(f"Crystal 2 A matrix:\n{A2}")
        print()
        
        # Calculate the relative rotation between the two orientations
        # R = A2 * A1^-1 gives the rotation from crystal1 to crystal2
        A1_inv = np.linalg.inv(A1)
        R = np.dot(A2, A1_inv)
        
        print("Relative Rotation Matrix (A2 * A1^-1):")
        print(f"{R}")
        print()
        
        # Calculate misorientation angle using trace of rotation matrix
        # For rotation matrix R, cos(theta) = (trace(R) - 1) / 2
        trace_R = np.trace(R)
        cos_theta = (trace_R - 1) / 2
        
        # Clamp to valid range for arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        
        print(f"RMS Misorientation Angle: {theta_deg:.4f}°")
        print()
        
        if theta_deg < 1.0:
            print("✅ SUCCESS: Orientations are very consistent (< 1°)")
            print("   Reference-based indexing appears to be working!")
        elif theta_deg < 5.0:
            print("✅ GOOD: Orientations are reasonably consistent (< 5°)")
            print("   Reference-based indexing is helping!")
        else:
            print("❌ ISSUE: Large misorientation detected")
            print("   Reference-based indexing may not be working properly")
            
        return theta_deg
        
    except Exception as e:
        print(f"Error checking orientations: {e}")
        return None

if __name__ == "__main__":
    check_orientations()