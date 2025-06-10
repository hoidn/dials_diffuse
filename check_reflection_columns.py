#!/usr/bin/env python3

import sys
sys.path.append('src')

import numpy as np

print("CHECKING WHAT Q-VECTORS ARE ALREADY AVAILABLE IN REFLECTION TABLE")
print("="*70)

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

experiment = ExperimentListFactory.from_json_file("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt")[0]
reflections = flex.reflection_table.from_file("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl")

print(f"Available columns in reflection table:")
for col in reflections.keys():
    print(f"  {col}")

print(f"\nChecking if DIALS already calculated q-vectors...")

# Check if we have rlp (reciprocal lattice point) or other q-vector columns
if 'rlp' in reflections:
    print(f"Found 'rlp' column (reciprocal lattice point)")
    rlp_0 = reflections['rlp'][0]
    print(f"  First rlp: {rlp_0}")
    
if 's1' in reflections:
    print(f"Found 's1' column (scattered beam vector)")
    s1_0 = np.array(reflections['s1'][0])
    s0_0 = np.array(experiment.beam.get_s0())
    q_from_s1 = s1_0 - s0_0
    print(f"  First s1: {s1_0}")
    print(f"  s0: {s0_0}")
    print(f"  q = s1 - s0: {q_from_s1} (mag: {np.linalg.norm(q_from_s1):.4f})")

# Calculate our reference q_pixel for comparison
hkl = reflections['miller_index'][0]
panel_id = int(reflections['panel'][0])
mm_pos = reflections['xyzcal.mm'][0]
x_mm, y_mm = mm_pos[0], mm_pos[1]

panel = experiment.detector[panel_id]
px_coords = panel.millimeter_to_pixel((x_mm, y_mm))

s0 = np.array(experiment.beam.get_s0())
k_magnitude = np.linalg.norm(s0)
P_lab = np.array(panel.get_pixel_lab_coord(px_coords))
D_scattered = P_lab
D_scattered_norm = np.linalg.norm(D_scattered)
s1_lab = D_scattered / D_scattered_norm
k_out = s1_lab * k_magnitude
q_pixel = k_out - s0

print(f"\nFor comparison:")
print(f"  Our calculated q_pixel: {q_pixel} (mag: {np.linalg.norm(q_pixel):.4f})")
print(f"  HKL: {hkl}")

# Check if s1 from table matches our calculation
if 's1' in reflections:
    s1_table = np.array(reflections['s1'][0])
    print(f"  s1 from table: {s1_table}")
    print(f"  s1 we calculated: {s1_lab * k_magnitude}")
    print(f"  s1 difference: {np.linalg.norm(s1_table - s1_lab * k_magnitude):.6f}")
    
    # Maybe DIALS s1 - s0 is already the right q-vector!
    q_from_dials = s1_table - s0
    print(f"  q from DIALS (s1-s0): {q_from_dials} (mag: {np.linalg.norm(q_from_dials):.4f})")
    
    print(f"  |q_pixel - q_dials|: {np.linalg.norm(q_pixel - q_from_dials):.6f}")