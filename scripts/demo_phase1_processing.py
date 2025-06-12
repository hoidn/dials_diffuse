#!/usr/bin/env python3
"""
Demo script for Phase 1 stills processing and validation.

This script demonstrates how to run and check the output of the stills processing,
orientation checking, and indexing validation components.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusepipe.crystallography.still_processing_and_validation import (
    StillProcessorAndValidatorComponent,
    create_default_config,
    create_default_extraction_config,
)
from diffusepipe.masking.pixel_mask_generator import (
    PixelMaskGenerator,
    create_default_static_params,
    create_default_dynamic_params,
)
from diffusepipe.masking.bragg_mask_generator import (
    BraggMaskGenerator,
    create_default_bragg_mask_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_still_processing_with_validation(
    image_path: str,
    external_pdb_path: str = None,
    output_dir: str = "phase1_demo_output",
    verbose: bool = False,
):
    """
    Demonstrate complete stills processing with geometric validation.

    Args:
        image_path: Path to CBF image file
        external_pdb_path: Optional path to reference PDB file
        output_dir: Directory for output files and plots
        verbose: Enable verbose logging
    """
    print("=" * 60)
    print("PHASE 1 STILLS PROCESSING AND VALIDATION DEMO")
    print("=" * 60)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Set up configurations
        print("\n1. Setting up configurations...")

        # Use PDB crystallographic parameters to help indexing
        # From 6o2h.pdb: P 1 space group, triclinic unit cell
        dials_config = create_default_config(
            enable_partiality=True,
            enable_shoeboxes=True,  # Enable for Option B Bragg masking
            known_unit_cell="27.424,32.134,34.513,88.66,108.46,111.88",
            known_space_group="P 1",  # Use full space group symbol
        )
        extraction_config = create_default_extraction_config()

        print(f"   ✓ DIALS config: partiality={dials_config.calculate_partiality}")
        print(f"   ✓ Validation tolerances:")
        print(f"     - Cell length: {extraction_config.cell_length_tol * 100:.1f}%")
        print(f"     - Cell angle: {extraction_config.cell_angle_tol}°")
        print(f"     - Orientation: {extraction_config.orient_tolerance_deg}°")
        print(
            f"     - Q-consistency: {extraction_config.q_consistency_tolerance_angstrom_inv} Å⁻¹"
        )

        # Step 2: Initialize processor
        print("\n2. Initializing still processor and validator...")
        processor = StillProcessorAndValidatorComponent()
        print(f"   ✓ Processor: {type(processor).__name__}")
        print(f"   ✓ Adapter: {type(processor.adapter).__name__}")
        print(f"   ✓ Validator: {type(processor.validator).__name__}")

        # Step 3: Process and validate the still
        print(f"\n3. Processing still image: {image_path}")
        if not Path(image_path).exists():
            print(f"   ⚠️  Image file not found: {image_path}")
            print(
                "   📝 This demo will show you the interface without actual processing"
            )
            print("   🔧 To run with real data, provide a valid CBF file path")
            return demo_interface_without_dials(output_dir)

        print("   🔄 Running DIALS stills_process...")
        print("   🔍 Performing geometric validation...")

        outcome = processor.process_and_validate_still(
            image_path=image_path,
            config=dials_config,
            extraction_config=extraction_config,
            external_pdb_path=external_pdb_path,
            output_dir=str(output_path),
        )

        # Step 4: Analyze results
        print(f"\n4. Processing Results:")
        print(f"   Status: {outcome.status}")
        print(f"   Message: {outcome.message}")

        if outcome.status == "SUCCESS":
            print("   ✅ DIALS processing and validation successful!")
            analyze_successful_outcome(outcome, output_path)

        elif outcome.status == "FAILURE_GEOMETRY_VALIDATION":
            print("   ⚠️  DIALS processing succeeded but validation failed!")
            analyze_validation_failure(outcome, output_path)

        else:
            print("   ❌ DIALS processing failed!")
            analyze_processing_failure(outcome, output_path)

        # Step 5: Demonstrate mask generation
        if outcome.status in ["SUCCESS", "FAILURE_GEOMETRY_VALIDATION"]:
            print(f"\n5. Demonstrating mask generation...")
            demo_mask_generation(outcome, output_path)

        print(f"\n📁 Check output files in: {output_path}")
        return outcome

    except ImportError as e:
        print(f"\n❌ Missing DIALS installation: {e}")
        print("   🔧 Install DIALS to run with real crystallographic data")
        return demo_interface_without_dials(output_dir)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.exception("Processing failed")
        return None


def demo_interface_without_dials(output_dir: str):
    """
    Demonstrate the interface and show what the output would look like,
    without requiring DIALS installation.
    """
    print("\n🔄 Running interface demonstration (no DIALS required)...")

    # Create mock outcome to show the interface
    from diffusepipe.types.types_IDL import OperationOutcome
    from diffusepipe.crystallography.still_processing_and_validation import (
        ValidationMetrics,
    )

    # Mock successful outcome
    metrics = ValidationMetrics()
    metrics.q_consistency_passed = True
    metrics.mean_delta_q_mag = 0.005
    metrics.max_delta_q_mag = 0.012
    metrics.num_reflections_tested = 1247
    metrics.pdb_cell_passed = True
    metrics.pdb_orientation_passed = True
    metrics.misorientation_angle_vs_pdb = 1.2

    mock_outcome = OperationOutcome(
        status="SUCCESS",
        message="Successfully processed and validated still image (DEMO)",
        error_code=None,
        output_artifacts={
            "experiment": "Mock_Experiment_Object",
            "reflections": "Mock_Reflections_Object",
            "validation_passed": True,
            "validation_metrics": metrics.to_dict(),
            "log_messages": "Mock DIALS processing completed successfully",
        },
    )

    print("   ✅ Mock processing successful!")
    analyze_successful_outcome(mock_outcome, Path(output_dir))

    # Create a mock validation failure example
    print("\n🔄 Example of validation failure...")
    metrics_fail = ValidationMetrics()
    metrics_fail.q_consistency_passed = False
    metrics_fail.mean_delta_q_mag = 0.025  # Above tolerance
    metrics_fail.max_delta_q_mag = 0.08
    metrics_fail.num_reflections_tested = 856

    mock_fail_outcome = OperationOutcome(
        status="FAILURE_GEOMETRY_VALIDATION",
        message="Geometric validation failed (DEMO)",
        error_code="GEOMETRY_VALIDATION_FAILED",
        output_artifacts={
            "experiment": "Mock_Experiment_Object",
            "reflections": "Mock_Reflections_Object",
            "validation_passed": False,
            "validation_metrics": metrics_fail.to_dict(),
            "log_messages": "DIALS processing successful, validation failed",
        },
    )

    print("   ⚠️  Mock validation failure!")
    analyze_validation_failure(mock_fail_outcome, Path(output_dir))

    return mock_outcome


def analyze_successful_outcome(outcome, output_path: Path):
    """Analyze and display results from successful processing."""
    artifacts = outcome.output_artifacts
    validation_metrics = artifacts.get("validation_metrics", {})

    print("   📊 Validation Results:")
    print(
        f"     • Q-consistency: {'✅ PASSED' if validation_metrics.get('q_consistency_passed') else '❌ FAILED'}"
    )

    mean_delta = validation_metrics.get("mean_delta_q_mag")
    max_delta = validation_metrics.get("max_delta_q_mag")
    if mean_delta is not None or max_delta is not None:
        mean_str = f"{mean_delta:.4f}" if mean_delta is not None else "N/A"
        max_str = f"{max_delta:.4f}" if max_delta is not None else "N/A"
        print(f"     • Mean |Δq|: {mean_str} Å⁻¹")
        print(f"     • Max |Δq|: {max_str} Å⁻¹")
        print(
            f"     • Reflections tested: {validation_metrics.get('num_reflections_tested', 'N/A')}"
        )

    if validation_metrics.get("pdb_cell_passed") is not None:
        print(
            f"     • PDB cell check: {'✅ PASSED' if validation_metrics.get('pdb_cell_passed') else '❌ FAILED'}"
        )
        print(
            f"     • PDB orientation: {'✅ PASSED' if validation_metrics.get('pdb_orientation_passed') else '❌ FAILED'}"
        )
        if validation_metrics.get("misorientation_angle_vs_pdb"):
            print(
                f"     • Misorientation angle: {validation_metrics['misorientation_angle_vs_pdb']:.2f}°"
            )

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed report
    report_path = output_path / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("STILLS PROCESSING AND VALIDATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Status: {outcome.status}\n")
        f.write(f"Message: {outcome.message}\n\n")
        f.write("Validation Metrics:\n")
        for key, value in validation_metrics.items():
            f.write(f"  {key}: {value}\n")

    print(f"   📄 Detailed report saved: {report_path}")


def analyze_validation_failure(outcome, output_path: Path):
    """Analyze and display results from validation failure."""
    artifacts = outcome.output_artifacts
    validation_metrics = artifacts.get("validation_metrics", {})

    print("   📊 Validation Failure Analysis:")

    # Check which validations failed
    failures = []
    if validation_metrics.get("q_consistency_passed") is False:
        failures.append("Q-vector consistency")
        print(f"     ❌ Q-consistency failed:")
        mean_delta = validation_metrics.get("mean_delta_q_mag")
        max_delta = validation_metrics.get("max_delta_q_mag")
        mean_str = f"{mean_delta:.4f}" if mean_delta is not None else "N/A"
        max_str = f"{max_delta:.4f}" if max_delta is not None else "N/A"
        print(f"        - Mean |Δq|: {mean_str} Å⁻¹")
        print(f"        - Max |Δq|: {max_str} Å⁻¹")

    if validation_metrics.get("pdb_cell_passed") is False:
        failures.append("PDB cell parameters")
        print(f"     ❌ PDB cell parameters failed")

    if validation_metrics.get("pdb_orientation_passed") is False:
        failures.append("PDB orientation")
        print(f"     ❌ PDB orientation failed")

    print(f"   🔧 Failed checks: {', '.join(failures) if failures else 'Unknown'}")
    print(f"   💡 This still should be excluded from subsequent processing")

    # Save failure report
    failure_path = output_path / "validation_failure_report.txt"
    with open(failure_path, "w") as f:
        f.write("VALIDATION FAILURE REPORT\n")
        f.write("=" * 25 + "\n\n")
        f.write(f"Status: {outcome.status}\n")
        f.write(f"Error Code: {outcome.error_code}\n")
        f.write(f"Message: {outcome.message}\n\n")
        f.write("Failed Validations:\n")
        for failure in failures:
            f.write(f"  - {failure}\n")
        f.write("\nValidation Metrics:\n")
        for key, value in validation_metrics.items():
            f.write(f"  {key}: {value}\n")

    print(f"   📄 Failure report saved: {failure_path}")


def analyze_processing_failure(outcome, output_path: Path):
    """Analyze and display results from DIALS processing failure."""
    print("   📊 DIALS Processing Failure:")
    print(f"     Error Code: {outcome.error_code}")
    print(f"     Message: {outcome.message}")

    if outcome.output_artifacts and "log_messages" in outcome.output_artifacts:
        print(f"     DIALS Log: {outcome.output_artifacts['log_messages']}")

    print(f"   💡 Check DIALS configuration and input image quality")

    # Save processing failure report
    failure_path = output_path / "processing_failure_report.txt"
    with open(failure_path, "w") as f:
        f.write("DIALS PROCESSING FAILURE REPORT\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Status: {outcome.status}\n")
        f.write(f"Error Code: {outcome.error_code}\n")
        f.write(f"Message: {outcome.message}\n\n")
        if outcome.output_artifacts:
            f.write("Output Artifacts:\n")
            for key, value in outcome.output_artifacts.items():
                f.write(f"  {key}: {value}\n")

    print(f"   📄 Failure report saved: {failure_path}")


def demo_mask_generation(outcome, output_path: Path):
    """Demonstrate mask generation pipeline."""
    print("   🎭 Generating masks for this still...")

    try:
        # Initialize mask generators
        pixel_generator = PixelMaskGenerator()
        bragg_generator = BraggMaskGenerator()

        # Get experiment and reflections from outcome
        artifacts = outcome.output_artifacts
        experiment = artifacts.get("experiment")
        reflections = artifacts.get("reflections")

        if isinstance(experiment, str) or isinstance(reflections, str):
            print("   📝 Using mock objects for mask generation demo")
            print("   🔧 With real DIALS data, masks would be generated here")
            return

        # Generate masks (this would work with real DIALS objects)
        print("   🔄 Would generate pixel masks...")
        print("   🔄 Would generate Bragg masks...")
        print("   🔄 Would combine into total masks...")

        print("   ✅ Mask generation pipeline ready!")

    except Exception as e:
        print(f"   ⚠️  Mask generation demo skipped: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Demo Phase 1 stills processing and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with test data
  python demo_phase1_processing.py --image tests/data/test_still.cbf
  
  # Run with PDB validation
  python demo_phase1_processing.py --image data/still.cbf --pdb data/reference.pdb
  
  # Run interface demo (no DIALS required)
  python demo_phase1_processing.py --demo-mode
  
  # Run with existing processed data
  python demo_phase1_processing.py --image lys_nitr_10_6_0491_dials_processing/imported.cbf
        """,
    )

    parser.add_argument("--image", help="Path to CBF still image file")
    parser.add_argument("--pdb", help="Path to reference PDB file for validation")
    parser.add_argument(
        "--output-dir",
        default="phase1_demo_output",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run interface demonstration without requiring DIALS",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.demo_mode:
        print("🎮 Running in demo mode (no DIALS required)")
        demo_interface_without_dials(args.output_dir)
    elif args.image:
        demo_still_processing_with_validation(
            image_path=args.image,
            external_pdb_path=args.pdb,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        # Default: run with any available test data or demo mode
        test_paths = [
            "lys_nitr_10_6_0491_dials_processing/imported.cbf",
            "lys_nitr_8_2_0110_dials_processing/imported.cbf",
            "tests/data/test_still.cbf",
            "747/test.cbf",
        ]

        found_data = False
        for test_path in test_paths:
            if Path(test_path).exists():
                print(f"🔍 Found test data: {test_path}")
                demo_still_processing_with_validation(
                    image_path=test_path,
                    external_pdb_path=args.pdb,
                    output_dir=args.output_dir,
                    verbose=args.verbose,
                )
                found_data = True
                break

        if not found_data:
            print("📝 No test data found, running interface demo...")
            demo_interface_without_dials(args.output_dir)


if __name__ == "__main__":
    main()
