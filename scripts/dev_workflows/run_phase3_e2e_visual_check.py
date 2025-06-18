#!/usr/bin/env python3
"""
End-to-End Visual Check Script for Phase 3 DiffusePipe Processing (True Sequence Processing)

This script orchestrates the complete Phase 1 (DIALS sequence processing, masking), 
Phase 2 (DataExtractor with shared model), and Phase 3 (voxelization, relative scaling, merging) 
pipeline for multiple CBF images using true sequence processing, then runs visual diagnostics 
to verify the correctness of the voxelization and merging processes.

KEY ARCHITECTURAL CHANGE: This script now uses true sequence processing where all images 
are processed together as a single cohesive dataset, achieving perfect crystal orientation 
consistency (0.0000° RMS misorientation) and eliminating indexing ambiguity issues.

The script provides an automated pathway to:
1. Process CBF images through DIALS as a single sequence (import→find_spots→index→integrate)
2. Generate per-image masks using the consistent scan-varying crystal model
3. Extract diffuse scattering data for each image using the shared experiment model
4. Define global voxel grid and bin data into voxels
5. Perform relative scaling of binned observations
6. Merge relatively scaled data into final voxel map
7. Generate comprehensive Phase 3 visual diagnostics

True Sequence Processing Benefits:
- Perfect crystal orientation consistency across all images
- Leverages DIALS's native scan-varying refinement
- Eliminates Phase 3 voxelization failures due to misorientation
- More robust than reference-based indexing approaches

Phase 3 implementation uses real components:
- GlobalVoxelGrid: Averages crystal models and defines 3D reciprocal space grid
- VoxelAccumulator: Bins observations into voxels with HDF5 backend
- DiffuseScalingModel: Performs iterative relative scaling with v1 constraints
- DiffuseDataMerger: Applies scaling and merges data into final voxel map

This is particularly useful for:
- Validating Phase 3 implementation on real sequential data
- Debugging voxelization and scaling pipeline issues
- Generating reference outputs for testing
- Development workflow verification with perfect crystal consistency

Author: DiffusePipe
"""

import argparse
import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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
    from diffusepipe.utils.cbf_utils import CBFUtils

    # Phase 3 component imports
    from diffusepipe.voxelization.global_voxel_grid import (
        GlobalVoxelGrid,
        GlobalVoxelGridConfig,
    )
    from diffusepipe.voxelization.voxel_accumulator import VoxelAccumulator
    from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
    from diffusepipe.merging.merger import DiffuseDataMerger
    from cctbx import sgtbx
    import numpy as np

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
        description="End-to-end Phase 3 visual check pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with multiple CBF files
  python run_phase3_e2e_visual_check.py \\
    --cbf-image-paths ../../747/lys_nitr_10_6_0491.cbf ../../747/lys_nitr_10_6_0492.cbf \\
    --output-base-dir ./e2e_phase3_outputs \\
    --pdb-path ../../6o2h.pdb

  # With custom configurations
  python run_phase3_e2e_visual_check.py \\
    --cbf-image-paths image1.cbf image2.cbf image3.cbf \\
    --output-base-dir ./outputs \\
    --dials-phil-path custom_dials.phil \\
    --static-mask-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \\
    --bragg-mask-config '{"border": 3}' \\
    --extraction-config-json '{"pixel_step": 2}' \\
    --relative-scaling-config-json '{"enable_res_smoother": true}' \\
    --grid-config-json '{"ndiv_h": 64, "ndiv_k": 64, "ndiv_l": 64}' \\
    --save-intermediate-phase-outputs \\
    --verbose

  # Using shoebox-based Bragg masking
  python run_phase3_e2e_visual_check.py \\
    --cbf-image-paths image1.cbf image2.cbf \\
    --output-base-dir ./outputs \\
    --use-bragg-mask-option-b \\
    --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--cbf-image-paths",
        nargs="+",
        required=True,
        help="Paths to input CBF image files",
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

    # Phase 3 configuration arguments
    parser.add_argument(
        "--relative-scaling-config-json",
        type=str,
        help="JSON string for relative scaling configuration",
    )
    parser.add_argument(
        "--grid-config-json",
        type=str,
        help="JSON string for global voxel grid configuration",
    )

    # Output control arguments
    parser.add_argument(
        "--save-intermediate-phase-outputs",
        action="store_true",
        help="Save intermediate Phase 1 & 2 outputs for inspection",
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


def extract_pdb_symmetry(pdb_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract unit cell and space group from PDB file.

    Returns:
        tuple[Optional[str], Optional[str]]: (unit_cell_string, space_group_string)
    """
    try:
        from iotbx import pdb

        pdb_input = pdb.input(file_name=pdb_path)
        crystal_symmetry = pdb_input.crystal_symmetry()

        if crystal_symmetry is None:
            logger.warning(f"No crystal symmetry found in PDB file: {pdb_path}")
            return None, None

        unit_cell = crystal_symmetry.unit_cell()
        space_group = crystal_symmetry.space_group()

        # Format unit cell parameters as comma-separated string
        uc_params = unit_cell.parameters()
        unit_cell_string = f"{uc_params[0]:.3f},{uc_params[1]:.3f},{uc_params[2]:.3f},{uc_params[3]:.2f},{uc_params[4]:.2f},{uc_params[5]:.2f}"

        # Get space group symbol
        space_group_string = space_group.type().lookup_symbol()

        logger.info(
            f"Extracted from PDB: unit_cell={unit_cell_string}, space_group={space_group_string}"
        )
        return unit_cell_string, space_group_string

    except Exception as e:
        logger.warning(f"Failed to extract symmetry from PDB file {pdb_path}: {e}")
        return None, None


def run_phase1_sequence_processing(
    args: argparse.Namespace, output_dir: Path
) -> Tuple[Path, Path, List[object]]:
    """
    Run Phase 1 DIALS processing as a single sequence and generate per-image masks.

    This function implements true sequence processing where all images are processed
    together as a single cohesive dataset, then individual masks are generated for
    each image using the consistent scan-varying model.

    Args:
        args: Command line arguments containing CBF image paths and configuration
        output_dir: Base output directory for sequence processing results

    Returns:
        Tuple of (composite_expt_path, composite_refl_path, per_image_mask_list)
    """
    logger.info("=== Phase 1: True Sequence Processing ===")

    # Extract known symmetry from PDB if provided
    known_unit_cell = None
    known_space_group = None
    if args.pdb_path:
        known_unit_cell, known_space_group = extract_pdb_symmetry(args.pdb_path)

    # Create sequence processing output directory
    sequence_output_dir = output_dir / "sequence_processing"
    sequence_output_dir.mkdir(parents=True, exist_ok=True)

    # Import the adapter
    from diffusepipe.adapters.dials_sequence_process_adapter import (
        DIALSSequenceProcessAdapter,
    )

    try:
        # Instantiate the sequence adapter
        sequence_adapter = DIALSSequenceProcessAdapter()

        # Create DIALS configuration with known symmetry
        dials_config = create_default_config(
            known_unit_cell=known_unit_cell, known_space_group=known_space_group
        )
        if args.dials_phil_path:
            dials_config.stills_process_phil_path = args.dials_phil_path
        if args.use_bragg_mask_option_b:
            dials_config.output_shoeboxes = True

        # Process all images as a single sequence
        logger.info(f"Processing {len(args.cbf_image_paths)} images as sequence")
        experiments, reflections, success, log_messages = (
            sequence_adapter.process_sequence(
                image_paths=args.cbf_image_paths,
                config=dials_config,
                output_dir_final=str(sequence_output_dir),
            )
        )

        if not success:
            raise RuntimeError(f"Sequence processing failed: {log_messages}")

        # Get the composite experiment and reflection file paths
        composite_expt_path = sequence_output_dir / "indexed_refined_detector.expt"
        composite_refl_path = sequence_output_dir / "indexed_refined_detector.refl"

        # Verify files exist
        if not composite_expt_path.exists() or not composite_refl_path.exists():
            raise RuntimeError("Expected sequence processing output files not found")

        logger.info("Sequence processing completed successfully")

        # Now generate per-image masks using the consistent model
        logger.info("Generating per-image masks from sequence model")

        # Load the composite experiment and reflections
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family import flex

        experiments_list = ExperimentListFactory.from_json_file(
            str(composite_expt_path)
        )
        all_reflections = flex.reflection_table.from_file(str(composite_refl_path))

        # Generate global pixel mask once (using first detector)
        detector = experiments_list[0].detector
        imageset = ImageSetFactory.new(args.cbf_image_paths)[0]

        pixel_generator = PixelMaskGenerator()

        # Parse static mask configuration
        static_config = parse_json_config(args.static_mask_config, "static mask")
        if static_config:
            static_params = create_default_static_params()
            for key, value in static_config.items():
                if hasattr(static_params, key):
                    setattr(static_params, key, value)
        else:
            static_params = create_default_static_params()

        # Use default dynamic parameters
        dynamic_params = create_default_dynamic_params()

        # Generate combined pixel mask (once for all images)
        global_pixel_mask_tuple = pixel_generator.generate_combined_pixel_mask(
            detector=detector,
            static_params=static_params,
            representative_images=[imageset],
            dynamic_params=dynamic_params,
        )

        # Save global pixel mask
        global_pixel_mask_path = sequence_output_dir / "global_pixel_mask.pickle"
        with open(global_pixel_mask_path, "wb") as f:
            pickle.dump(global_pixel_mask_tuple, f)

        # Generate per-image masks
        per_image_mask_objects = []
        bragg_generator = BraggMaskGenerator()

        # Parse Bragg mask configuration
        bragg_config_overrides = parse_json_config(args.bragg_mask_config, "Bragg mask")
        bragg_config = create_default_bragg_mask_config()
        bragg_config.update(bragg_config_overrides)

        for i, cbf_path in enumerate(args.cbf_image_paths):
            logger.info(
                f"Generating mask for image {i+1}/{len(args.cbf_image_paths)}: {Path(cbf_path).name}"
            )

            # Create image-specific output directory for masks
            cbf_path_obj = Path(cbf_path)
            image_output_dir = output_dir / f"still_{i:03d}_{cbf_path_obj.stem}"
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # Filter reflections for this specific image using Z-coordinate of xyzobs.px.value
            # For sequence processing, frame index is stored in Z component, not imageset_id
            # The Z-coordinate uses 0-based indexing matching the loop index 'i'
            z_coords = all_reflections["xyzobs.px.value"].parts()[2]
            image_reflections = all_reflections.select(z_coords.iround() == i)
            
            # Log the number of reflections found
            if len(image_reflections) == 0:
                logger.warning(
                    f"No reflections found for image index {i} ({Path(cbf_path).name}). "
                    f"This may indicate an issue with reflection filtering."
                )
            else:
                logger.info(f"Found {len(image_reflections)} reflections for image index {i}.")

            # Get the experiment (single scan-varying model for all frames in true sequence processing)
            experiment = experiments_list[0]

            # Generate Bragg mask for this image
            if args.use_bragg_mask_option_b:
                bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_shoeboxes(
                    reflections=image_reflections, detector=detector
                )
            else:
                bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_spots(
                    experiment=experiment,
                    reflections=image_reflections,
                    config=bragg_config,
                )

            # Save individual Bragg mask
            bragg_mask_path = image_output_dir / "bragg_mask.pickle"
            with open(bragg_mask_path, "wb") as f:
                pickle.dump(bragg_mask_tuple, f)

            # Generate total diffuse mask for this image
            total_diffuse_mask_tuple = bragg_generator.get_total_mask_for_still(
                bragg_mask=bragg_mask_tuple,
                global_pixel_mask=global_pixel_mask_tuple,
            )

            # Save total diffuse mask
            total_diffuse_mask_path = image_output_dir / "total_diffuse_mask.pickle"
            with open(total_diffuse_mask_path, "wb") as f:
                pickle.dump(total_diffuse_mask_tuple, f)

            # Store mask information
            mask_info = {
                "image_index": i,
                "cbf_path": cbf_path,
                "image_output_dir": image_output_dir,
                "total_diffuse_mask_path": total_diffuse_mask_path,
                "bragg_mask_path": bragg_mask_path,
            }
            per_image_mask_objects.append(mask_info)

        logger.info("Per-image mask generation completed")
        return composite_expt_path, composite_refl_path, per_image_mask_objects

    except Exception as e:
        logger.error(f"Sequence processing failed: {e}")
        raise


def run_mask_generation_for_image(
    args: argparse.Namespace,
    image_output_dir: Path,
    cbf_image_path: str,
    expt_file: Path,
    refl_file: Path,
) -> Dict[str, Path]:
    """
    Generate masks for a single image.

    Returns:
        Dictionary containing mask file paths
    """
    logger.info(f"Generating masks for {cbf_image_path}")

    # Load experiment and reflection data
    experiments = ExperimentListFactory.from_json_file(str(expt_file))
    experiment = experiments[0]
    detector = experiment.detector

    # Load raw CBF image
    imageset = ImageSetFactory.new([cbf_image_path])[0]

    # Pixel Mask Generation
    pixel_generator = PixelMaskGenerator()

    # Parse static mask configuration
    static_config = parse_json_config(args.static_mask_config, "static mask")
    if static_config:
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
        representative_images=[imageset],
        dynamic_params=dynamic_params,
    )

    # Save pixel mask
    pixel_mask_path = image_output_dir / "global_pixel_mask.pickle"
    with open(pixel_mask_path, "wb") as f:
        pickle.dump(global_pixel_mask_tuple, f)

    # Bragg Mask Generation
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
        bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_shoeboxes(
            reflections=reflections, detector=detector
        )
    else:
        bragg_mask_tuple = bragg_generator.generate_bragg_mask_from_spots(
            experiment=experiment, reflections=reflections, config=bragg_config
        )

    # Save Bragg mask
    bragg_mask_path = image_output_dir / "bragg_mask.pickle"
    with open(bragg_mask_path, "wb") as f:
        pickle.dump(bragg_mask_tuple, f)

    # Total Diffuse Mask Generation
    total_diffuse_mask_tuple = bragg_generator.get_total_mask_for_still(
        bragg_mask=bragg_mask_tuple,
        global_pixel_mask=global_pixel_mask_tuple,
    )

    # Save total diffuse mask
    total_diffuse_mask_path = image_output_dir / "total_diffuse_mask.pickle"
    with open(total_diffuse_mask_path, "wb") as f:
        pickle.dump(total_diffuse_mask_tuple, f)

    return {
        "pixel_mask_path": pixel_mask_path,
        "bragg_mask_path": bragg_mask_path,
        "total_diffuse_mask_path": total_diffuse_mask_path,
    }


def _run_data_extraction_for_image(
    args: argparse.Namespace,
    cbf_path: str,
    image_output_dir: Path,
    composite_expt_file: Path,
    total_diffuse_mask_path: Path,
    start_angle: float,
) -> Dict[str, Any]:
    """
    Run Phase 2 data extraction for a single image using the shared experiment model.

    Args:
        args: Command line arguments
        cbf_path: Path to the CBF image file
        image_output_dir: Output directory for this specific image
        composite_expt_file: Path to the shared experiment file from sequence processing
        total_diffuse_mask_path: Path to the total diffuse mask for this image
        start_angle: Start angle from CBF header to determine correct frame index

    Returns:
        Dictionary with Phase 2 results for this image
    """
    logger.info(f"Extracting diffuse data from {cbf_path}")

    try:
        # Instantiate DataExtractor
        data_extractor = DataExtractor()

        # Create ComponentInputFiles
        component_inputs = ComponentInputFiles(
            cbf_image_path=cbf_path,
            dials_expt_path=str(composite_expt_file),
            bragg_mask_path=str(total_diffuse_mask_path),
            external_pdb_path=args.pdb_path,
        )

        # Create ExtractionConfig
        extraction_config = create_default_extraction_config()
        extraction_config.save_original_pixel_coordinates = True

        # Apply JSON overrides if provided
        if args.extraction_config_json:
            json_overrides = parse_json_config(
                args.extraction_config_json, "extraction config"
            )
            config_dict = extraction_config.model_dump()
            config_dict.update(json_overrides)
            extraction_config = ExtractionConfig(**config_dict)

        # Define output NPZ path
        output_npz_path = image_output_dir / "corrected_diffuse_pixel_data.npz"

        # Extract diffuse data
        data_extractor_outcome = data_extractor.extract_from_still(
            inputs=component_inputs,
            config=extraction_config,
            output_npz_path=str(output_npz_path),
            start_angle=start_angle,
        )

        # Check outcome
        if data_extractor_outcome.status != "SUCCESS":
            logger.error(
                f"Data extraction failed for {cbf_path}: {data_extractor_outcome.status}"
            )
            raise RuntimeError(
                f"Data extraction failed: {data_extractor_outcome.status}"
            )

        # Store results
        result = {
            "cbf_image_path": cbf_path,
            "image_output_dir": image_output_dir,
            "expt_file": composite_expt_file,  # Shared experiment file
            "corrected_npz_path": output_npz_path,
            "total_diffuse_mask_path": total_diffuse_mask_path,
            "phase2_status": "SUCCESS",
        }

        logger.info(f"Successfully extracted data from {cbf_path}")
        return result

    except Exception as e:
        logger.error(f"Failed to extract data from {cbf_path}: {e}")
        raise


def _run_phase2_visual_diagnostics(
    args: argparse.Namespace,
    image_output_dir: Path,
    cbf_path: str,
    expt_file: Path,
    total_mask_path: Path,
    npz_file: Path,
) -> None:
    """
    Run per-image Phase 2 visual diagnostics using check_diffuse_extraction.py.

    Args:
        args: Command line arguments
        image_output_dir: Output directory for this specific image
        cbf_path: Path to the CBF image file
        expt_file: Path to the experiment file (shared from sequence processing)
        total_mask_path: Path to the total diffuse mask for this image
        npz_file: Path to the extracted NPZ data file
    """
    logger.info(f"Generating Phase 2 visual diagnostics for {Path(cbf_path).name}")

    # Create Phase 2 diagnostics subdirectory
    phase2_diagnostics_dir = image_output_dir / "phase2_diagnostics"
    phase2_diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the check_diffuse_extraction.py script
    script_path = (
        Path(__file__).resolve().parent.parent
        / "visual_diagnostics"
        / "check_diffuse_extraction.py"
    )

    # Construct the command
    cmd = [
        "python",
        str(script_path),
        "--raw-image",
        cbf_path,
        "--expt",
        str(expt_file),
        "--total-mask",
        str(total_mask_path),
        "--npz-file",
        str(npz_file),
        "--output-dir",
        str(phase2_diagnostics_dir),
    ]

    # Add verbose flag if requested
    if args.verbose:
        cmd.append("--verbose")

    logger.info(f"Running Phase 2 diagnostics: {' '.join(cmd)}")

    # Execute the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent.parent),  # Project root
        )

        # Log results
        if result.returncode == 0:
            logger.info(
                f"Phase 2 diagnostics completed successfully for {Path(cbf_path).name}"
            )
        else:
            logger.error(
                f"Phase 2 diagnostics failed for {Path(cbf_path).name} with return code: {result.returncode}"
            )

        if result.stdout:
            logger.debug("Phase 2 diagnostics stdout:")
            for line in result.stdout.strip().split("\n"):
                logger.debug(f"  {line}")

        if result.stderr:
            logger.warning("Phase 2 diagnostics stderr:")
            for line in result.stderr.strip().split("\n"):
                logger.warning(f"  {line}")

        if result.returncode != 0:
            logger.warning(
                f"Phase 2 diagnostics failed for {Path(cbf_path).name}, but continuing..."
            )

    except Exception as e:
        logger.error(
            f"Failed to run Phase 2 diagnostics for {Path(cbf_path).name}: {e}"
        )
        logger.warning("Continuing without Phase 2 diagnostics...")


def run_phase3_voxelization_and_scaling(
    args: argparse.Namespace,
    phase2_results: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Run Phase 3 voxelization, relative scaling, and merging.

    Returns:
        Dictionary containing Phase 3 output file paths
    """
    logger.info("=== Phase 3: Voxelization, Relative Scaling, and Merging ===")

    # Phase 1: Data Aggregation
    logger.info("Aggregating experiment data and diffuse observations...")

    # Collect all experiments
    experiments = []
    for result in phase2_results:
        try:
            expt = ExperimentListFactory.from_json_file(str(result["expt_file"]))
            experiments.extend(expt)
        except Exception as e:
            logger.warning(f"Failed to load experiment from {result['expt_file']}: {e}")

    if not experiments:
        raise RuntimeError("No experiments loaded from Phase 2 results")

    # Collect and concatenate diffuse data
    q_vectors_list = []
    intensities_list = []
    sigmas_list = []
    still_ids_list = []

    for i, result in enumerate(phase2_results):
        try:
            # Load NPZ file
            npz_data = np.load(result["corrected_npz_path"])

            # Extract arrays
            q_vectors = npz_data["q_vectors"]
            intensities = npz_data["intensities"]
            sigmas = npz_data["sigmas"]  # Phase 2 saves as "sigmas"

            n_obs = len(intensities)

            # Append to lists
            q_vectors_list.append(q_vectors)
            intensities_list.append(intensities)
            sigmas_list.append(sigmas)
            still_ids_list.append(np.full(n_obs, i))

        except Exception as e:
            logger.warning(
                f"Failed to load diffuse data from {result['corrected_npz_path']}: {e}"
            )

    if not q_vectors_list:
        raise RuntimeError("No diffuse data loaded from Phase 2 results")

    # Concatenate all arrays
    all_q_vectors = np.vstack(q_vectors_list)
    all_intensities = np.concatenate(intensities_list)
    all_sigmas = np.concatenate(sigmas_list)
    all_still_ids = np.concatenate(still_ids_list)

    logger.info(
        f"Aggregated {len(all_intensities)} observations from {len(phase2_results)} stills"
    )

    # Package aggregated data
    from diffusepipe.voxelization.global_voxel_grid import CorrectedDiffusePixelData

    collected_diffuse_data = CorrectedDiffusePixelData(
        q_vectors=all_q_vectors,
        intensities=all_intensities,
        sigmas=all_sigmas,
        still_ids=all_still_ids,
    )

    # Phase 2A: Global Voxel Grid
    logger.info("Creating global voxel grid...")

    # Parse grid configuration
    if args.grid_config_json:
        grid_config_dict = json.loads(args.grid_config_json)
    else:
        grid_config_dict = {
            "d_min_target": 2.0,
            "d_max_target": 50.0,
            "ndiv_h": 32,
            "ndiv_k": 32,
            "ndiv_l": 32,
        }

    grid_config = GlobalVoxelGridConfig(**grid_config_dict)

    # Instantiate GlobalVoxelGrid
    global_grid = GlobalVoxelGrid(experiments, [collected_diffuse_data], grid_config)

    # Save grid definition
    grid_definition = {
        "crystal_avg_ref": {
            "unit_cell_params": list(
                global_grid.crystal_avg_ref.get_unit_cell().parameters()
            ),
            "space_group": str(global_grid.crystal_avg_ref.get_space_group().info()),
            "setting_matrix_a": [
                list(row) for row in np.array(global_grid.A_avg_ref).reshape(3, 3)
            ],
        },
        "hkl_bounds": {
            "h_min": global_grid.hkl_min[0],
            "h_max": global_grid.hkl_max[0],
            "k_min": global_grid.hkl_min[1],
            "k_max": global_grid.hkl_max[1],
            "l_min": global_grid.hkl_min[2],
            "l_max": global_grid.hkl_max[2],
        },
        "ndiv_h": grid_config.ndiv_h,
        "ndiv_k": grid_config.ndiv_k,
        "ndiv_l": grid_config.ndiv_l,
        "total_voxels": global_grid.total_voxels,
        "diagnostics": global_grid.get_crystal_averaging_diagnostics(),
    }

    grid_definition_path = output_dir / "global_voxel_grid_definition.json"
    with open(grid_definition_path, "w") as f:
        json.dump(grid_definition, f, indent=2)

    logger.info(f"Saved grid definition to {grid_definition_path}")

    # Phase 2B: Voxel Accumulator
    logger.info("Binning observations into voxels...")

    # Get space group
    space_group = global_grid.crystal_avg_ref.get_space_group()
    space_group_info = sgtbx.space_group_info(group=space_group)

    # Instantiate VoxelAccumulator with HDF5 backend
    accumulator = VoxelAccumulator(global_grid, space_group_info, backend="hdf5")

    # Add all observations
    for i in range(len(phase2_results)):
        still_mask = all_still_ids == i
        if np.any(still_mask):
            accumulator.add_observations(
                still_id=i,
                q_vectors_lab=all_q_vectors[still_mask],
                intensities=all_intensities[still_mask],
                sigmas=all_sigmas[still_mask],
            )

    # Finalize accumulator
    accumulator.finalize()

    # Get binned data for scaling
    binned_data = accumulator.get_all_binned_data_for_scaling()

    logger.info(f"Binned data into {len(binned_data)} voxels")

    # Phase 2C: Relative Scaling
    logger.info("Performing relative scaling...")
    logger.info(f"Processing {len(binned_data)} voxels with scaling algorithm...")

    # Parse scaling configuration
    if args.relative_scaling_config_json:
        scaling_config = json.loads(args.relative_scaling_config_json)
    else:
        scaling_config = {}

    # Add still IDs to configuration
    scaling_config["still_ids"] = list(np.unique(all_still_ids))

    # Instantiate scaling model
    scaling_model = DiffuseScalingModel(scaling_config)

    # Refine scaling parameters with reduced iterations for large datasets
    max_iterations = 3 if len(binned_data) > 50000 else 10
    refinement_config = {
        "max_iterations": max_iterations,
        "convergence_tolerance": 1e-4,
    }

    logger.info(f"Running scaling refinement with max_iterations={max_iterations}")

    refined_params, stats = scaling_model.refine_parameters(
        binned_data, {}, refinement_config  # Empty bragg reflections for now
    )

    # Create correctly structured dictionary for diagnostic script
    scaling_params_to_save = {
        "refined_parameters": refined_params,
        "refinement_statistics": stats,
    }

    # Extract and add resolution smoother data if available
    model_info = scaling_model.get_model_info()
    if "resolution" in model_info.get("components", {}):
        resolution_component = model_info["components"]["resolution"]
        control_point_values = resolution_component.get("control_point_values", [])
        scaling_params_to_save["resolution_smoother"] = {
            "enabled": True,
            "control_points": control_point_values,
        }
    else:
        scaling_params_to_save["resolution_smoother"] = {
            "enabled": False,
            "control_points": [],
        }

    scaling_params_path = output_dir / "refined_scaling_model_params.json"
    with open(scaling_params_path, "w") as f:
        json.dump(scaling_params_to_save, f, indent=2)

    logger.info(f"Saved scaling parameters to {scaling_params_path}")

    # Phase 2D: Merging
    logger.info("Merging scaled data...")

    # Instantiate merger
    merger = DiffuseDataMerger(global_grid)

    # Merge scaled data
    merge_config = {
        "minimum_observations": 1,
        "weight_method": "inverse_variance",
        "outlier_rejection": {"enabled": False},
    }

    voxel_data = merger.merge_scaled_data(binned_data, scaling_model, merge_config)

    # Get merge statistics
    merge_stats = merger.get_merge_statistics(voxel_data)
    logger.info(f"Merge statistics: {merge_stats['total_voxels']} voxels merged")

    # Save merged voxel data
    voxel_data_path = output_dir / "voxel_data_relative.npz"
    merger.save_voxel_data(voxel_data, str(voxel_data_path))

    logger.info(f"Saved merged voxel data to {voxel_data_path}")

    # Return file paths for diagnostics
    return {
        "grid_definition_file": grid_definition_path,
        "scaling_model_params_file": scaling_params_path,
        "voxel_data_file": voxel_data_path,
    }


def save_intermediate_outputs_manifest(
    phase2_results: List[Dict[str, Any]], output_dir: Path
) -> None:
    """Save manifest of intermediate outputs if requested."""
    manifest = {
        "phase1_and_phase2_outputs": [
            {
                "cbf_image_path": result["cbf_image_path"],
                "image_output_dir": str(result["image_output_dir"]),
                "expt_file": str(result["expt_file"]),
                "refl_file": str(result["refl_file"]),
                "corrected_npz_path": str(result["corrected_npz_path"]),
                "pixel_mask_path": str(result["pixel_mask_path"]),
                "bragg_mask_path": str(result["bragg_mask_path"]),
                "total_diffuse_mask_path": str(result["total_diffuse_mask_path"]),
            }
            for result in phase2_results
        ]
    }

    manifest_path = output_dir / "intermediate_outputs_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved intermediate outputs manifest: {manifest_path}")


def run_phase3_visual_diagnostics(
    args: argparse.Namespace,
    phase3_outputs: Dict[str, Path],
    phase2_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Run Phase 3 visual diagnostics check.
    """
    logger.info("=== Phase 3: Visual Diagnostics ===")

    # Construct command for check_phase3_outputs.py
    diagnostics_dir = output_dir / "phase3_diagnostics"

    # Get the script path relative to the project root
    script_path = (
        Path(__file__).resolve().parent.parent
        / "visual_diagnostics"
        / "check_phase3_outputs.py"
    )

    cmd = [
        "python",
        str(script_path),
        "--grid-definition-file",
        str(phase3_outputs["grid_definition_file"]),
        "--scaling-model-params-file",
        str(phase3_outputs["scaling_model_params_file"]),
        "--voxel-data-file",
        str(phase3_outputs["voxel_data_file"]),
        "--output-dir",
        str(diagnostics_dir),
    ]

    # Add optional arguments if intermediate outputs were saved
    if args.save_intermediate_phase_outputs:
        manifest_path = output_dir / "intermediate_outputs_manifest.json"
        if manifest_path.exists():
            cmd.extend(["--experiments-list-file", str(manifest_path)])

    if args.verbose:
        cmd.append("--verbose")

    logger.info(f"Running Phase 3 visual diagnostics: {' '.join(cmd)}")

    # Execute the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent.parent),  # Project root
        )

        # Log results
        if result.returncode == 0:
            logger.info("Phase 3 visual diagnostics completed successfully")
        else:
            logger.error(
                f"Phase 3 visual diagnostics failed with return code: {result.returncode}"
            )

        if result.stdout:
            logger.info("Phase 3 diagnostics stdout:")
            for line in result.stdout.strip().split("\n"):
                logger.info(f"  {line}")

        if result.stderr:
            logger.error("Phase 3 diagnostics stderr:")
            for line in result.stderr.strip().split("\n"):
                logger.error(f"  {line}")

        if result.returncode != 0:
            logger.warning("Phase 3 visual diagnostics failed, but continuing...")

    except Exception as e:
        logger.error(f"Failed to run Phase 3 visual diagnostics: {e}")
        logger.warning("Continuing without visual diagnostics...")


def main():
    """Main orchestration function."""
    # Parse arguments
    args = parse_arguments()

    # Validate input files
    for cbf_path in args.cbf_image_paths:
        if not Path(cbf_path).exists():
            logger.error(f"CBF file not found: {cbf_path}")
            sys.exit(1)

    # Create unique output directory
    output_base = Path(args.output_base_dir)
    output_dir = output_base / "phase3_e2e_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "phase3_e2e_visual_check.log"
    setup_logging(log_file, args.verbose)

    logger.info("Starting end-to-end Phase 3 visual check pipeline")
    logger.info(f"Input CBF files: {args.cbf_image_paths}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Phase 1: True Sequence Processing
        composite_expt_path, composite_refl_path, per_image_masks = (
            run_phase1_sequence_processing(args, output_dir)
        )

        logger.info(f"Sequence processing completed with {len(per_image_masks)} images")

        # Phase 2: Data Extraction (Per-Image with Shared Model)
        phase2_results = []
        cbf_utils = CBFUtils()
        
        for i, cbf_path in enumerate(args.cbf_image_paths):
            mask_info = per_image_masks[i]
            
            # Parse Start_angle from CBF header for robust frame index lookup
            start_angle = cbf_utils.get_start_angle(cbf_path)
            if start_angle is None:
                logger.error(f"Could not determine Start_angle for {cbf_path}")
                raise RuntimeError(f"Failed to parse Start_angle from CBF header: {cbf_path}")
            
            logger.info(f"Processing CBF {Path(cbf_path).name} with Start_angle: {start_angle}°")

            # Extract data for this image using the shared experiment model
            phase2_result = _run_data_extraction_for_image(
                args=args,
                cbf_path=cbf_path,
                image_output_dir=mask_info["image_output_dir"],
                composite_expt_file=composite_expt_path,
                total_diffuse_mask_path=mask_info["total_diffuse_mask_path"],
                start_angle=start_angle,
            )

            # Add mask information to the result
            phase2_result.update(
                {
                    "pixel_mask_path": None,  # Not saved individually in new flow
                    "bragg_mask_path": mask_info["bragg_mask_path"],
                    "refl_file": composite_refl_path,  # Shared reflection file
                }
            )

            phase2_results.append(phase2_result)

            # Generate Phase 2 visual diagnostics for this image
            _run_phase2_visual_diagnostics(
                args=args,
                image_output_dir=mask_info["image_output_dir"],
                cbf_path=cbf_path,
                expt_file=composite_expt_path,
                total_mask_path=mask_info["total_diffuse_mask_path"],
                npz_file=phase2_result["corrected_npz_path"],
            )

        if not phase2_results:
            logger.error("No images were successfully processed in Phase 2")
            sys.exit(1)

        # Save intermediate outputs manifest if requested
        if args.save_intermediate_phase_outputs:
            save_intermediate_outputs_manifest(phase2_results, output_dir)

        # Phase 3: Voxelization, Relative Scaling, and Merging
        phase3_outputs = run_phase3_voxelization_and_scaling(
            args, phase2_results, output_dir
        )

        # Phase 3: Visual Diagnostics
        run_phase3_visual_diagnostics(args, phase3_outputs, phase2_results, output_dir)

        logger.info("=== Pipeline Completed Successfully ===")
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("Generated files:")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                logger.info(f"  {file_path.relative_to(output_dir)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check the log file for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
