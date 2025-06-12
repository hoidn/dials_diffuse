#!/usr/bin/env python3

import sys

sys.path.append("src")

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

# Enable debug logging for validation
logger = logging.getLogger(
    "diffusepipe.crystallography.still_processing_and_validation"
)
logger.setLevel(logging.DEBUG)

print("Testing if we can import and run the validation...")

try:
    from diffusepipe.crystallography.still_processing_and_validation import (
        ModelValidator,
    )
    from diffusepipe.types.types_IDL import ExtractionConfig

    print("✓ Imports successful")

    # Try to create validator
    validator = ModelValidator()
    print("✓ Validator created")

    # Test if we can access the processed data directly
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.array_family import flex

    print("Loading experiment from processed data...")
    experiment = ExperimentListFactory.from_json_file(
        "lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt"
    )[0]
    reflections = flex.reflection_table.from_file(
        "lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl"
    )

    print(f"Loaded experiment with {len(reflections)} reflections")
    print("Testing the patched _check_q_consistency method directly...")

    # This should trigger our debug logging
    ok, stats = validator._check_q_consistency(experiment, reflections, tolerance=0.05)

    print(f"Direct validation result: {ok}")
    print(f"Stats: {stats}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
