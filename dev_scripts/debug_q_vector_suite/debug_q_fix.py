#!/usr/bin/env python3

"""
Debug script to test Q-vector coordinate frame transformation
and compare with the consistency_checker.py approach.
"""

import sys

sys.path.append("src")

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
import numpy as np
from scitbx import matrix


def test_q_transformation(expt_file, refl_file):
    """Test Q-vector transformation using both methods."""
    print(f"Testing Q-vector transformation with {expt_file}")

    # Load data
    experiments = ExperimentListFactory.from_json_file(expt_file)
    reflections = flex.reflection_table.from_file(refl_file)
    experiment = experiments[0]

    print(f"Loaded {len(reflections)} reflections")

    # Test first few reflections
    n_test = min(5, len(reflections))
    print(f"Testing first {n_test} reflections:")

    for i in range(n_test):
        hkl = reflections["miller_index"][i]
        print(f"\nReflection {i}: HKL = {hkl}")

        # Method 1: Direct crystal method (problematic)
        try:
            q_crystal_method = experiment.crystal.hkl_to_reciprocal_space_vec(hkl)
            q_crystal = np.array(q_crystal_method.elems)
            print(
                f"  Crystal method: {q_crystal} (mag: {np.linalg.norm(q_crystal):.4f})"
            )
        except AttributeError:
            q_crystal = None
            print("  Crystal method: Not available")

        # Method 2: Lab frame transformation (fixed)
        hkl_vec = matrix.col(hkl)
        A = matrix.sqr(experiment.crystal.get_A())
        S = matrix.sqr(experiment.goniometer.get_setting_rotation())
        F = matrix.sqr(experiment.goniometer.get_fixed_rotation())
        C = matrix.sqr((1, 0, 0, 0, 0, -1, 0, 1, 0))
        R_lab = C * S * F
        q_lab_method = R_lab * A * hkl_vec
        q_lab = np.array(q_lab_method.elems)
        print(f"  Lab method:     {q_lab} (mag: {np.linalg.norm(q_lab):.4f})")

        # Method 3: Just A*hkl (no rotation)
        A_array = np.array(A.elems).reshape(3, 3)
        hkl_array = np.array(hkl)
        q_A_only = A_array @ hkl_array
        print(f"  A*hkl only:     {q_A_only} (mag: {np.linalg.norm(q_A_only):.4f})")

        # Compare methods
        if q_crystal is not None:
            diff_crystal_lab = np.linalg.norm(q_crystal - q_lab)
            diff_crystal_A = np.linalg.norm(q_crystal - q_A_only)
            print(f"  |crystal - lab|: {diff_crystal_lab:.4f}")
            print(f"  |crystal - A*hkl|: {diff_crystal_A:.4f}")

        diff_lab_A = np.linalg.norm(q_lab - q_A_only)
        print(f"  |lab - A*hkl|:   {diff_lab_A:.4f}")

        # Calculate q_pixel for comparison
        if "xyzcal.mm" in reflections:
            try:
                panel_id = int(reflections["panel"][i])
                mm_pos = reflections["xyzcal.mm"][i]
                x_mm, y_mm = mm_pos[0], mm_pos[1]

                panel = experiment.detector[panel_id]
                px_coords = panel.millimeter_to_pixel((x_mm, y_mm))

                # Calculate q_pixel using the same method as in consistency checker
                s0 = np.array(experiment.beam.get_s0())
                k_magnitude = np.linalg.norm(s0)
                P_lab = np.array(panel.get_pixel_lab_coord(px_coords))
                D_scattered = P_lab
                D_scattered_norm = np.linalg.norm(D_scattered)
                s1_lab = D_scattered / D_scattered_norm
                k_out = s1_lab * k_magnitude
                q_pixel = k_out - s0

                print(
                    f"  q_pixel:        {q_pixel} (mag: {np.linalg.norm(q_pixel):.4f})"
                )
                print(f"  |lab - pixel|:   {np.linalg.norm(q_lab - q_pixel):.4f}")
                if q_crystal is not None:
                    print(
                        f"  |crystal - pixel|: {np.linalg.norm(q_crystal - q_pixel):.4f}"
                    )

            except Exception as e:
                print(f"  q_pixel calculation failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING Q-VECTOR COORDINATE TRANSFORMATION")
    print("=" * 60)

    # Test both datasets
    print("\n1. Testing WORKING dataset (lys_nitr_8_2_0110):")
    test_q_transformation(
        "lys_nitr_8_2_0110_dials_processing/indexed_refined_detector.expt",
        "lys_nitr_8_2_0110_dials_processing/indexed_refined_detector.refl",
    )

    print("\n" + "=" * 60)
    print("\n2. Testing PROBLEMATIC dataset (lys_nitr_10_6_0491):")
    test_q_transformation(
        "lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt",
        "lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl",
    )
