#!/usr/bin/env python3
"""
End-to-End Visual Check Script for Phase 2 DiffusePipe Processing

This script orchestrates the complete Phase 1 (DIALS processing, masking) and 
Phase 2 (DataExtractor) pipeline for a single CBF image, then runs visual 
diagnostics to verify the correctness of the diffuse scattering extraction 
and correction processes.

The script provides an automated pathway to:
1. Process a CBF image through DIALS (import, find spots, index, refine)
2. Generate pixel masks (static + dynamic)
3. Generate Bragg masks (from spots or shoeboxes)
4. Combine masks into total diffuse mask
5. Extract diffuse scattering data with pixel corrections
6. Generate comprehensive visual diagnostics

This is particularly useful for:
- Validating Phase 2 implementation on real data
- Debugging extraction pipeline issues
- Generating reference outputs for testing
- Development workflow verification

Author: DiffusePipe
"""

import argparse
import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    # DIALS/DXTBX imports
    from dxtbx.imageset import ImageSetFactory
    from dxtbx.model.experiment_list import ExperimentListFactory

    # Project component imports
    from diffusepipe.crystallography.still_processing_and_validation import (
        StillProcessorAndValidatorComponent,
        create_default_config,
        create_default_extraction_config,
    )
    from diffusepipe.types.types_IDL import (
        ExtractionConfig,
        ComponentInputFiles,
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
    from diffusepipe.extraction.data_extractor import DataExtractor

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(
        "Please ensure DIALS is properly installed and the project is set up correctly."
    )
    sys.exit(1)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end Phase 2 visual check pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with a CBF file
  python run_phase2_e2e_visual_check.py \\
    --cbf-image ../../747/lys_nitr_10_6_0491.cbf \\
    --output-base-dir ./e2e_outputs \\
    --pdb-path ../../6o2h.pdb

  # With custom configurations
  python run_phase2_e2e_visual_check.py \\
    --cbf-image image.cbf \\
    --output-base-dir ./outputs \\
    --dials-phil-path custom_dials.phil \\
    --static-mask-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \\
    --bragg-mask-config '{"border": 3}' \\
    --extraction-config-json '{"pixel_step": 2}' \\
    --verbose

  # Using shoebox-based Bragg masking
  python run_phase2_e2e_visual_check.py \\
    --cbf-image image.cbf \\
    --output-base-dir ./outputs \\
    --use-bragg-mask-option-b \\
    --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--cbf-image", type=str, required=True, help="Path to input CBF image file"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        required=True,
        help="Base output directory (unique subdirectory will be created)",
    )

    # Optional DIALS/processing arguments
    parser.add_argument(
        "--dials-phil-path",
        type=str,
        help="Path to custom DIALS PHIL configuration file",
    )
    parser.add_argument(
        "--pdb-path", type=str, help="Path to external PDB file for validation"
    )

    # Masking configuration arguments
    parser.add_argument(
        "--static-mask-config",
        type=str,
        help="JSON string for static mask configuration (StaticMaskParams)",
    )
    parser.add_argument(
        "--bragg-mask-config", type=str, help="JSON string for Bragg mask configuration"
    )
    parser.add_argument(
        "--use-bragg-mask-option-b",
        action="store_true",
        help="Use shoebox-based Bragg masking (requires shoeboxes from DIALS)",
    )

    # Extraction configuration arguments
    parser.add_argument(
        "--extraction-config-json",
        type=str,
        help="JSON string for extraction configuration overrides",
    )
    parser.add_argument(
        "--pixel-step", type=int, help="Pixel sampling step size for extraction"
    )
    parser.add_argument(
        "--save-pixel-coords",
        action="store_true",
        default=True,
        help="Save original pixel coordinates in NPZ output (default: True)",
    )

    # General arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_logging(log_file_path: Path, verbose: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])

    logger.info(f"Logging configured. Log file: {log_file_path}")


def parse_json_config(json_str: Optional[str], config_name: str) -> Dict[str, Any]:
    """Parse JSON configuration string."""
    if not json_str:
        return {}

    try:
        config = json.loads(json_str)
        logger.info(f"Parsed {config_name} configuration: {config}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {config_name} JSON: {e}")
        raise


def run_phase1_dials_processing(
    args: argparse.Namespace, image_output_dir: Path
) -> tuple[Path, Path]:
    """
    Run Phase 1 DIALS processing and validation.

    Returns:
        tuple[Path, Path]: Paths to experiment file and reflection file
    """
    logger.info("=== Phase 1: DIALS Processing and Validation ===")

    # 2.1: DIALS Processing & Validation
    logger.info("Initializing DIALS processor...")
    processor = StillProcessorAndValidatorComponent()

    # Create DIALS configuration
    dials_config = create_default_config()
    if args.dials_phil_path:
        dials_config.stills_process_phil_path = args.dials_phil_path
    if args.use_bragg_mask_option_b:
        dials_config.output_shoeboxes = True

    # Create extraction config for validation
    extraction_config = create_default_extraction_config()

    logger.info(f"Processing CBF image: {args.cbf_image}")
    logger.info(f"Output directory: {image_output_dir}")

    # Process the still
    still_outcome = processor.process_and_validate_still(
        image_path=args.cbf_image,
        config=dials_config,
        extraction_config=extraction_config,
        external_pdb_path=args.pdb_path,
        output_dir=str(image_output_dir),
    )

    # Check outcome
    if still_outcome.status != "SUCCESS":
        logger.error(f"DIALS processing failed: {still_outcome.status}")
        logger.error(f"Error details: {still_outcome.message}")
        raise RuntimeError(f"Phase 1 DIALS processing failed: {still_outcome.message}")

    logger.info("DIALS processing completed successfully")

    # Identify output files
    expt_file = image_output_dir / "indexed_refined_detector.expt"
    refl_file = image_output_dir / "indexed_refined_detector.refl"

    # Verify files exist
    if not expt_file.exists():
        raise FileNotFoundError(f"Expected experiment file not found: {expt_file}")
    if not refl_file.exists():
        raise FileNotFoundError(f"Expected reflection file not found: {refl_file}")

    logger.info(f"Generated experiment file: {expt_file}")
    logger.info(f"Generated reflection file: {refl_file}")

    return expt_file, refl_file


def run_phase1_mask_generation(
    args: argparse.Namespace, image_output_dir: Path, expt_file: Path, refl_file: Path
) -> tuple[Path, Path, Path]:
    """
    Run Phase 1 mask generation (pixel, Bragg, and total masks).

    Returns:
        tuple[Path, Path, Path]: Paths to pixel mask, Bragg mask, and total diffuse mask
    """
    logger.info("=== Phase 1: Mask Generation ===")

    # Load experiment and reflection data
    logger.info("Loading experiment and reflection data...")
    experiments = ExperimentListFactory.from_json_file(str(expt_file))
    experiment = experiments[0]
    detector = experiment.detector

    # Load raw CBF image
    imageset = ImageSetFactory.new([args.cbf_image])[0]

    # 2.2: Pixel Mask Generation
    logger.info("Generating pixel masks...")
    pixel_generator = PixelMaskGenerator()

    # Parse static mask configuration
    static_config = parse_json_config(args.static_mask_config, "static mask")
    if static_config:
        # Update default with provided config
        static_params = create_default_static_params()
        for key, value in static_config.items():
            if hasattr(static_params, key):
                setattr(static_params, key, value)
    else:
        static_params = create_default_static_params()

    # Use default dynamic parameters
    dynamic_params = create_default_dynamic_params()

    # Generate combined pixel mask
    global_pixel_mask_tuple = pixel_generator.generate_combined_pixel_mask(
        detector=detector,
        static_params=static_params,
        imagesets=[imageset],
        dynamic_params=dynamic_params,
    )

    # Save pixel mask
    pixel_mask_path = image_output_dir / "global_pixel_mask.pickle"
    with open(pixel_mask_path, "wb") as f:
        pickle.dump(global_pixel_mask_tuple, f)
    logger.info(f"Saved pixel mask: {pixel_mask_path}")

    # 2.3: Bragg Mask Generation
    logger.info("Generating Bragg mask...")
    bragg_generator = BraggMaskGenerator()

    # Parse Bragg mask configuration
    bragg_config_overrides = parse_json_config(args.bragg_mask_config, "Bragg mask")
    bragg_config = create_default_bragg_mask_config()
    bragg_config.update(bragg_config_overrides)

    # Load reflection data
    from dials.array_family import flex

    reflections = flex.reflection_table.from_file(str(refl_file))

    # Generate Bragg mask
    if args.use_bragg_mask_option_b:
        logger.info("Using shoebox-based Bragg masking (Option B)")
        bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_shoeboxes(
            reflections=reflections, detector=detector
        )
    else:
        logger.info("Using spot-based Bragg masking (Option A)")
        bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_spots(
            experiment=experiment, reflections=reflections, config=bragg_config
        )

    # Save Bragg mask
    bragg_mask_path = image_output_dir / "bragg_mask.pickle"
    with open(bragg_mask_path, "wb") as f:
        pickle.dump(bragg_mask_tuple, f)
    logger.info(f"Saved Bragg mask: {bragg_mask_path}")

    # 2.4: Total Diffuse Mask Generation
    logger.info("Generating total diffuse mask...")
    total_diffuse_mask_tuple = bragg_generator.get_total_mask_for_still(
        bragg_mask_tuple=bragg_mask_tuple,
        global_pixel_mask_tuple=global_pixel_mask_tuple,
    )

    # Save total diffuse mask
    total_diffuse_mask_path = image_output_dir / "total_diffuse_mask.pickle"
    with open(total_diffuse_mask_path, "wb") as f:
        pickle.dump(total_diffuse_mask_tuple, f)
    logger.info(f"Saved total diffuse mask: {total_diffuse_mask_path}")

    return pixel_mask_path, bragg_mask_path, total_diffuse_mask_path


def run_phase2_data_extraction(
    args: argparse.Namespace,
    image_output_dir: Path,
    expt_file: Path,
    total_diffuse_mask_path: Path,
) -> Path:
    """
    Run Phase 2 data extraction.

    Returns:
        Path: Path to output NPZ file
    """
    logger.info("=== Phase 2: Data Extraction ===")

    # 3.1: Instantiate DataExtractor
    data_extractor = DataExtractor()

    # 3.2: Create ComponentInputFiles
    component_inputs = ComponentInputFiles(
        cbf_image_path=args.cbf_image,
        dials_expt_path=str(expt_file),
        bragg_mask_path=str(total_diffuse_mask_path),  # Use total diffuse mask
        external_pdb_path=args.pdb_path,
    )

    # 3.3: Create ExtractionConfig
    extraction_config = create_default_extraction_config()

    # Set pixel coordinate saving
    extraction_config.save_original_pixel_coordinates = args.save_pixel_coords

    # Apply JSON overrides if provided
    if args.extraction_config_json:
        json_overrides = parse_json_config(
            args.extraction_config_json, "extraction config"
        )
        # Update config with overrides
        config_dict = extraction_config.model_dump()
        config_dict.update(json_overrides)
        extraction_config = ExtractionConfig(**config_dict)

    # Apply pixel step if provided
    if args.pixel_step:
        extraction_config.pixel_step = args.pixel_step

    logger.info(f"Extraction configuration: {extraction_config}")

    # 3.4: Define output NPZ path
    output_npz_path = image_output_dir / "diffuse_data.npz"

    # 3.5: Extract diffuse data
    logger.info(f"Extracting diffuse data to: {output_npz_path}")
    data_extractor_outcome = data_extractor.extract_from_still(
        component_inputs=component_inputs,
        extraction_config=extraction_config,
        output_npz_path=str(output_npz_path),
    )

    # 3.6: Check outcome
    if data_extractor_outcome.status != "SUCCESS":
        logger.error(f"Data extraction failed: {data_extractor_outcome.status}")
        logger.error(f"Error details: {data_extractor_outcome.message}")
        raise RuntimeError(
            f"Phase 2 data extraction failed: {data_extractor_outcome.message}"
        )

    logger.info("Data extraction completed successfully")
    return output_npz_path


def run_visual_check(
    args: argparse.Namespace,
    image_output_dir: Path,
    expt_file: Path,
    total_diffuse_mask_path: Path,
    bragg_mask_path: Path,
    pixel_mask_path: Path,
    npz_file: Path,
) -> None:
    """
    Run visual diagnostics check.
    """
    logger.info("=== Phase 3: Visual Diagnostics ===")

    # 4.1: Construct command for check_diffuse_extraction.py
    diagnostics_dir = image_output_dir / "extraction_diagnostics"

    # Get the script path relative to the project root
    script_path = (
        Path(__file__).resolve().parent.parent
        / "visual_diagnostics"
        / "check_diffuse_extraction.py"
    )

    cmd = [
        "python",
        str(script_path),
        "--raw-image",
        args.cbf_image,
        "--expt",
        str(expt_file),
        "--total-mask",
        str(total_diffuse_mask_path),
        "--npz-file",
        str(npz_file),
        "--bragg-mask",
        str(bragg_mask_path),
        "--pixel-mask",
        str(pixel_mask_path),
        "--output-dir",
        str(diagnostics_dir),
    ]

    if args.verbose:
        cmd.append("--verbose")

    logger.info(f"Running visual diagnostics: {' '.join(cmd)}")

    # 4.2: Execute the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent.parent),  # Project root
        )

        # 4.3: Log results
        if result.returncode == 0:
            logger.info("Visual diagnostics completed successfully")
        else:
            logger.error(
                f"Visual diagnostics failed with return code: {result.returncode}"
            )

        if result.stdout:
            logger.info("Visual diagnostics stdout:")
            for line in result.stdout.strip().split("\n"):
                logger.info(f"  {line}")

        if result.stderr:
            logger.error("Visual diagnostics stderr:")
            for line in result.stderr.strip().split("\n"):
                logger.error(f"  {line}")

        if result.returncode != 0:
            logger.warning("Visual diagnostics failed, but continuing...")

    except Exception as e:
        logger.error(f"Failed to run visual diagnostics: {e}")
        logger.warning("Continuing without visual diagnostics...")


def main():
    """Main orchestration function."""
    # Parse arguments
    args = parse_arguments()

    # Create unique output directory
    cbf_path = Path(args.cbf_image)
    output_base = Path(args.output_base_dir)
    image_output_dir = output_base / cbf_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = image_output_dir / "e2e_visual_check.log"
    setup_logging(log_file, args.verbose)

    logger.info("Starting end-to-end Phase 2 visual check pipeline")
    logger.info(f"Input CBF: {args.cbf_image}")
    logger.info(f"Output directory: {image_output_dir}")

    try:
        # Phase 1: DIALS Processing
        expt_file, refl_file = run_phase1_dials_processing(args, image_output_dir)

        # Phase 1: Mask Generation
        pixel_mask_path, bragg_mask_path, total_diffuse_mask_path = (
            run_phase1_mask_generation(args, image_output_dir, expt_file, refl_file)
        )

        # Phase 2: Data Extraction
        npz_file = run_phase2_data_extraction(
            args, image_output_dir, expt_file, total_diffuse_mask_path
        )

        # Phase 3: Visual Diagnostics
        run_visual_check(
            args,
            image_output_dir,
            expt_file,
            total_diffuse_mask_path,
            bragg_mask_path,
            pixel_mask_path,
            npz_file,
        )

        logger.info("=== Pipeline Completed Successfully ===")
        logger.info(f"All outputs saved to: {image_output_dir}")
        logger.info("Generated files:")
        for file_path in sorted(image_output_dir.rglob("*")):
            if file_path.is_file():
                logger.info(f"  {file_path.relative_to(image_output_dir)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check the log file for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
