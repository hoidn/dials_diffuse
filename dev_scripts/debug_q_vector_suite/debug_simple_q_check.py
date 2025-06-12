#!/usr/bin/env python3
"""
Simple debug script to isolate the Q-vector validation issue.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")


def debug_q_consistency_simple():
    """Debug the Q-vector consistency check step by step."""
    print("=" * 60)
    print("SIMPLE Q-VECTOR DEBUG")
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

    experiment = outcome.output_artifacts.get("experiment")
    reflections = outcome.output_artifacts.get("reflections")

    print(f"\n1. Reflection table inspection:")
    print(f"   Type: {type(reflections)}")
    print(f"   Length: {len(reflections)}")

    # Check columns
    required_cols = ["miller_index", "panel"]
    for col in required_cols:
        present = col in reflections
        print(f"   {col}: {'✓ present' if present else '❌ missing'}")

    pos_cols = ["xyzcal.mm", "xyzobs.mm.value", "xyzcal.px", "xyzobs.px.value"]
    print(f"\n2. Position columns:")
    for col in pos_cols:
        present = col in reflections
        print(f"   {col}: {'✓ present' if present else '❌ missing'}")

    # Try to access the data
    print(f"\n3. Trying to access data:")
    try:
        n_total = len(reflections)
        print(f"   Total reflections: {n_total}")

        # Get first reflection data
        if n_total > 0:
            hkl = reflections["miller_index"][0]
            panel_id = reflections["panel"][0]

            print(f"   First reflection:")
            print(f"     Miller index: {hkl}")
            print(f"     Panel: {panel_id}")

            # Try position data
            if "xyzcal.mm" in reflections:
                pos_mm = reflections["xyzcal.mm"][0]
                print(f"     xyzcal.mm: {pos_mm}")

            # Test q_bragg calculation
            try:
                q_bragg_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl)
                q_bragg = np.array(q_bragg_scitbx.elems)
                print(f"     q_bragg: {q_bragg} (mag: {np.linalg.norm(q_bragg):.4f})")
            except Exception as e:
                print(f"     q_bragg calculation failed: {e}")

            # Test pixel coordinate conversion
            try:
                if "xyzcal.mm" in reflections:
                    mm_pos = reflections["xyzcal.mm"][0]
                    x_mm, y_mm = mm_pos[0], mm_pos[1]

                    panel = experiment.detector[int(round(panel_id))]
                    pixel_coords = panel.millimeter_to_pixel((x_mm, y_mm))
                    print(f"     mm to pixel: ({x_mm}, {y_mm}) -> {pixel_coords}")

                    # Test lab coordinates
                    lab_coord_scitbx = panel.get_pixel_lab_coord(pixel_coords)
                    lab_coord = np.array(lab_coord_scitbx.elems)
                    print(f"     lab coordinates: {lab_coord}")

                    # Test q_pixel calculation
                    s0 = np.array(experiment.beam.get_s0())
                    k_magnitude = np.linalg.norm(s0)
                    scattered_direction_norm = np.linalg.norm(lab_coord)
                    s1_unit = lab_coord / scattered_direction_norm
                    s1 = s1_unit * k_magnitude
                    q_pixel = s1 - s0

                    print(f"     s0: {s0}")
                    print(f"     s1: {s1}")
                    print(
                        f"     q_pixel: {q_pixel} (mag: {np.linalg.norm(q_pixel):.4f})"
                    )

                    # Calculate difference
                    delta_q = q_bragg - q_pixel
                    delta_q_mag = np.linalg.norm(delta_q)
                    print(f"     |Δq|: {delta_q_mag:.6f} Å⁻¹")

            except Exception as e:
                print(f"     Coordinate calculation failed: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"   Data access failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_q_consistency_simple()
