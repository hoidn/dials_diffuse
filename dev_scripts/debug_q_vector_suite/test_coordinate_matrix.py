#!/usr/bin/env python3

import sys
sys.path.append('src')

import numpy as np
from scitbx import matrix

print("TESTING DIFFERENT COORDINATE TRANSFORMATION APPROACHES")
print("="*60)

# Load one dataset to test with
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

experiment = ExperimentListFactory.from_json_file("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt")[0]
reflections = flex.reflection_table.from_file("lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl")

# Get first reflection for testing
hkl = reflections['miller_index'][0]
print(f"Testing with first reflection: HKL = {hkl}")

# Get matrices
A = matrix.sqr(experiment.crystal.get_A())
S = matrix.sqr(experiment.goniometer.get_setting_rotation())
F = matrix.sqr(experiment.goniometer.get_fixed_rotation())

print(f"A matrix: {A.elems}")
print(f"S matrix: {S.elems}")
print(f"F matrix: {F.elems}")

# Test different C matrices and transformations
test_cases = [
    ("Our current C", matrix.sqr((1,0,0, 0,0,-1, 0,1,0))),
    ("Identity (no C)", matrix.sqr((1,0,0, 0,1,0, 0,0,1))),
    ("Y/Z flip only", matrix.sqr((1,0,0, 0,0,1, 0,1,0))),
    ("Different C", matrix.sqr((1,0,0, 0,-1,0, 0,0,-1))),
]

hkl_vec = matrix.col(hkl)

for name, C in test_cases:
    print(f"\n{name}:")
    print(f"  C matrix: {C.elems}")
    
    # Test different transformation orders
    transformations = [
        ("C*S*F*A", C * S * F * A),
        ("S*F*C*A", S * F * C * A),
        ("A only", A),
        ("C*A only", C * A)
    ]
    
    for trans_name, trans_matrix in transformations:
        q_result = trans_matrix * hkl_vec
        q_array = np.array(q_result.elems)
        print(f"    {trans_name}: {q_array} (mag: {np.linalg.norm(q_array):.4f})")

# Test what the built-in DIALS method gives us (if available)
print(f"\nBuilt-in DIALS method:")
try:
    q_dials = experiment.crystal.hkl_to_reciprocal_space_vec(hkl)
    q_dials_array = np.array(q_dials.elems)
    print(f"  hkl_to_reciprocal_space_vec: {q_dials_array} (mag: {np.linalg.norm(q_dials_array):.4f})")
except AttributeError:
    print("  hkl_to_reciprocal_space_vec not available")

# Calculate q_pixel for comparison
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

print(f"\nq_pixel reference: {q_pixel} (mag: {np.linalg.norm(q_pixel):.4f})")