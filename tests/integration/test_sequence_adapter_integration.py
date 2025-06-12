#!/usr/bin/env python3
"""Test the new sequence-based DIALS adapter."""

import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffusepipe.adapters.dials_sequence_process_adapter import (
    DIALSSequenceProcessAdapter,
)
from diffusepipe.crystallography.still_processing_and_validation import (
    create_default_config,
)


def test_sequence_adapter():
    """Test the sequence adapter with the problematic CBF file."""

    cbf_path = "747/lys_nitr_10_6_0491.cbf"

    if not Path(cbf_path).exists():
        print(f"CBF file not found: {cbf_path}")
        assert False, f"CBF file not found: {cbf_path}"

    print(f"Testing sequence adapter with: {cbf_path}")

    try:
        # Set up configuration
        config = create_default_config(
            enable_partiality=True,
            enable_shoeboxes=True,
            known_unit_cell="27.424,32.134,34.513,88.66,108.46,111.88",
            known_space_group="P 1",
        )

        # Initialize adapter
        adapter = DIALSSequenceProcessAdapter()

        # Process the image
        print("Processing with sequence adapter...")
        experiment, reflections, success, log_messages = adapter.process_still(
            image_path=cbf_path, config=config
        )

        if success:
            print("✅ SEQUENCE ADAPTER SUCCESS!")
            print(f"Experiment: {experiment}")
            print(f"Reflections: {len(reflections) if reflections else 0}")
            print(f"Log messages: {log_messages}")
            assert success is True
        else:
            print("❌ Sequence adapter failed")
            print(f"Log: {log_messages}")
            assert success is True  # This will fail and raise an AssertionError

    except Exception as e:
        print(f"❌ Sequence adapter error: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Exception occurred: {e}"


if __name__ == "__main__":
    success = test_sequence_adapter()

    if success:
        print("\n🎉 SOLUTION FOUND!")
        print(
            "The sequence-based adapter successfully processes these 0.1° oscillation images."
        )
        print(
            "The issue was that stills_process doesn't work well with oscillation data."
        )
        print(
            "The manual workflow succeeds because it treats them as sequences, not stills."
        )
    else:
        print("\n❌ The sequence adapter also failed.")
        print("Further investigation needed.")
