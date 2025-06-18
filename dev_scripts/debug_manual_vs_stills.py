#!/usr/bin/env python3
"""
Debug script to compare manual DIALS workflow vs stills_process
"""

import tempfile
import subprocess
import sys
from pathlib import Path


def run_manual_workflow(cbf_path, output_dir):
    """Run the exact manual DIALS workflow that works"""
    print(f"=== MANUAL WORKFLOW in {output_dir} ===")

    # Change to output directory
    original_cwd = Path.cwd()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        import os

        os.chdir(output_path)

        # Step 1: Import
        print("Step 1: dials.import")
        result = subprocess.run(
            ["dials.import", str(Path(original_cwd) / cbf_path)],
            capture_output=True,
            text=True,
        )
        print(f"Import stdout: {result.stdout}")
        if result.stderr:
            print(f"Import stderr: {result.stderr}")

        # Step 2: Find spots
        print("Step 2: dials.find_spots")
        result = subprocess.run(
            [
                "dials.find_spots",
                "imported.expt",
                "spotfinder.filter.min_spot_size=3",
                "spotfinder.threshold.algorithm=dispersion",
            ],
            capture_output=True,
            text=True,
        )
        print(f"Find_spots stdout: {result.stdout}")
        if result.stderr:
            print(f"Find_spots stderr: {result.stderr}")

        # Step 3: Index with known symmetry
        print("Step 3: dials.index")
        result = subprocess.run(
            [
                "dials.index",
                "imported.expt",
                "strong.refl",
                'indexing.known_symmetry.space_group="P 1"',
                "indexing.known_symmetry.unit_cell=27.424,32.134,34.513,88.66,108.46,111.88",
                'output.experiments="indexed_manual.expt"',
                'output.reflections="indexed_manual.refl"',
            ],
            capture_output=True,
            text=True,
        )
        print(f"Index stdout: {result.stdout}")
        if result.stderr:
            print(f"Index stderr: {result.stderr}")

        # Check results
        if Path("indexed_manual.expt").exists():
            print("✅ Manual workflow succeeded!")
            return True
        else:
            print("❌ Manual workflow failed!")
            return False

    finally:
        os.chdir(original_cwd)


def run_stills_process_workflow(cbf_path, output_dir):
    """Run stills_process with our current configuration"""
    print(f"=== STILLS_PROCESS WORKFLOW in {output_dir} ===")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Add project src to path for testing
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from diffusepipe.crystallography.still_processing_and_validation import (
            StillProcessorAndValidatorComponent,
            create_default_config,
        )

        # Set up configuration
        dials_config = create_default_config(
            enable_partiality=True,
            enable_shoeboxes=True,
            known_unit_cell="27.424,32.134,34.513,88.66,108.46,111.88",
            known_space_group="P 1",
        )

        # Initialize processor
        processor = StillProcessorAndValidatorComponent()

        # Process the still using our adapter
        result = processor.adapter.process_still(
            image_path=cbf_path, config=dials_config
        )

        experiment, reflections, success, log_messages = result

        if success:
            print("✅ stills_process succeeded!")
            print(f"Experiment: {experiment}")
            print(f"Reflections: {len(reflections) if reflections else 0}")
            return True
        else:
            print("❌ stills_process failed!")
            print(f"Log: {log_messages}")
            return False

    except Exception as e:
        print(f"❌ stills_process error: {e}")
        return False


def main():
    cbf_path = "747/lys_nitr_10_6_0491.cbf"

    if not Path(cbf_path).exists():
        print(f"CBF file not found: {cbf_path}")
        return

    # Test manual workflow
    with tempfile.TemporaryDirectory() as temp_dir:
        manual_dir = Path(temp_dir) / "manual"
        manual_success = run_manual_workflow(cbf_path, manual_dir)

    # Test stills_process workflow
    with tempfile.TemporaryDirectory() as temp_dir:
        stills_dir = Path(temp_dir) / "stills"
        stills_success = run_stills_process_workflow(cbf_path, stills_dir)

    print("\n=== COMPARISON SUMMARY ===")
    print(f"Manual workflow: {'✅ SUCCESS' if manual_success else '❌ FAILED'}")
    print(f"stills_process:  {'✅ SUCCESS' if stills_success else '❌ FAILED'}")

    if manual_success and not stills_success:
        print("\n🔍 DIAGNOSIS: Manual workflow succeeds but stills_process fails")
        print("   This confirms there's a difference in the approaches.")
        print("   Need to identify what stills_process is doing differently.")
    elif not manual_success and not stills_success:
        print("\n🔍 DIAGNOSIS: Both workflows fail")
        print("   The issue may be with the data or environment setup.")
    elif manual_success and stills_success:
        print("\n🔍 DIAGNOSIS: Both workflows succeed")
        print("   The issue may have been resolved.")


if __name__ == "__main__":
    main()
