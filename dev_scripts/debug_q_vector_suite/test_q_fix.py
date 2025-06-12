#!/usr/bin/env python3
"""
Quick test of the Q-vector calculation fixes.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")


def test_q_vector_fixes():
    """Test that our Q-vector API fixes work."""
    print("=" * 60)
    print("TESTING Q-VECTOR CALCULATION FIXES")
    print("=" * 60)

    from diffusepipe.crystallography.still_processing_and_validation import (
        StillProcessorAndValidatorComponent,
        create_default_config,
        create_default_extraction_config,
    )

    # Process the image
    processor = StillProcessorAndValidatorComponent()
    config = create_default_config(
        enable_partiality=True,
        known_unit_cell="27.424,32.134,34.513,88.66,108.46,111.88",
        known_space_group="P 1",
    )
    extraction_config = create_default_extraction_config()

    outcome = processor.process_and_validate_still(
        image_path="747/lys_nitr_10_6_0491.cbf",
        config=config,
        extraction_config=extraction_config,
        external_pdb_path="6o2h.pdb",
    )

    print(f"\nProcessing outcome: {outcome.status}")
    print(f"Message: {outcome.message}")

    if outcome.output_artifacts:
        validation_metrics = outcome.output_artifacts.get("validation_metrics", {})
        print(f"\nValidation Results:")
        print(
            f"  Q-consistency passed: {validation_metrics.get('q_consistency_passed')}"
        )
        print(
            f"  Reflections tested: {validation_metrics.get('num_reflections_tested')}"
        )
        print(f"  Mean |Δq|: {validation_metrics.get('mean_delta_q_mag')}")
        print(f"  Max |Δq|: {validation_metrics.get('max_delta_q_mag')}")
        print(f"  Median |Δq|: {validation_metrics.get('median_delta_q_mag')}")

    return outcome


if __name__ == "__main__":
    test_q_vector_fixes()
