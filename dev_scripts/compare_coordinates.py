#!/usr/bin/env python3

import sys
sys.path.append('src')

import logging
logging.basicConfig(level=logging.INFO)

print("COMPARING COORDINATE TRANSFORMATIONS BETWEEN DATASETS")
print("="*60)

from diffusepipe.crystallography.still_processing_and_validation import ModelValidator

# Test both datasets with the same validator
datasets = [
    ("WORKING", "lys_nitr_8_2_0110_dials_processing"),
    ("PROBLEMATIC", "lys_nitr_10_6_0491_dials_processing")
]

for name, dataset_dir in datasets:
    print(f"\n{name} DATASET ({dataset_dir}):")
    print("-" * 50)
    
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family import flex
        
        experiment = ExperimentListFactory.from_json_file(f"{dataset_dir}/indexed_refined_detector.expt")[0]
        reflections = flex.reflection_table.from_file(f"{dataset_dir}/indexed_refined_detector.refl")
        
        validator = ModelValidator()
        
        # Test with higher tolerance to see actual errors
        ok, stats = validator._check_q_consistency(experiment, reflections, tolerance=0.5)
        
        print(f"Validation passed: {ok}")
        print(f"Mean |Δq|: {stats['mean']:.4f} Å⁻¹")
        print(f"Max |Δq|: {stats['max']:.4f} Å⁻¹")
        
        # Let's check the coordinate transformation matrices are identical
        from scitbx import matrix
        A = matrix.sqr(experiment.crystal.get_A())
        S = matrix.sqr(experiment.goniometer.get_setting_rotation())
        F = matrix.sqr(experiment.goniometer.get_fixed_rotation())
        C = matrix.sqr((1,0,0, 0,0,-1, 0,1,0))
        R_lab = C * S * F
        
        print(f"R_lab transformation matrix: {R_lab.elems}")
        
    except Exception as e:
        print(f"Error: {e}")