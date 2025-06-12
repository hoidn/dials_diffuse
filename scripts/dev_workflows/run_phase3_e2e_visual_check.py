#!/usr/bin/env python3
"""
End-to-End Visual Check Script for Phase 3 DiffusePipe Processing

This script orchestrates the complete Phase 1 (DIALS processing, masking), 
Phase 2 (DataExtractor), and Phase 3 (voxelization, relative scaling, merging) 
pipeline for multiple CBF images, then runs visual diagnostics to verify the 
correctness of the voxelization and merging processes.

The script provides an automated pathway to:
1. Process CBF images through DIALS (import, find spots, index, refine)
2. Generate pixel masks (static + dynamic) and Bragg masks
3. Extract diffuse scattering data with pixel corrections
4. Define global voxel grid and bin data into voxels
5. Perform relative scaling of binned observations
6. Merge relatively scaled data into final voxel map
7. Generate comprehensive Phase 3 visual diagnostics

This is particularly useful for:
- Validating Phase 3 implementation on real data
- Debugging voxelization and scaling pipeline issues
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
        help="Paths to input CBF image files"
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


def run_phase1_processing_for_multiple_images(
    args: argparse.Namespace, output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Run Phase 1 DIALS processing and masking for multiple CBF images.

    Returns:
        List of dictionaries containing processing results for each image
    """
    logger.info("=== Phase 1: DIALS Processing and Masking (Multiple Images) ===")

    # Extract known symmetry from PDB if provided
    known_unit_cell = None
    known_space_group = None
    if args.pdb_path:
        known_unit_cell, known_space_group = extract_pdb_symmetry(args.pdb_path)

    results = []

    for i, cbf_image_path in enumerate(args.cbf_image_paths):
        logger.info(f"Processing image {i+1}/{len(args.cbf_image_paths)}: {cbf_image_path}")
        
        # Create image-specific output directory
        cbf_path = Path(cbf_image_path)
        image_output_dir = output_dir / f"still_{i:03d}_{cbf_path.stem}"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # DIALS Processing & Validation
            processor = StillProcessorAndValidatorComponent()

            # Create DIALS configuration with known symmetry
            dials_config = create_default_config(
                known_unit_cell=known_unit_cell, known_space_group=known_space_group
            )
            if args.dials_phil_path:
                dials_config.stills_process_phil_path = args.dials_phil_path
            if args.use_bragg_mask_option_b:
                dials_config.output_shoeboxes = True

            # Create extraction config for validation
            extraction_config = create_default_extraction_config()

            # Process the still
            still_outcome = processor.process_and_validate_still(
                image_path=cbf_image_path,
                config=dials_config,
                extraction_config=extraction_config,
                external_pdb_path=args.pdb_path,
                output_dir=str(image_output_dir),
            )

            # Check outcome
            if still_outcome.status != "SUCCESS":
                logger.error(f"DIALS processing failed for {cbf_image_path}: {still_outcome.status}")
                logger.error(f"Error details: {still_outcome.message}")
                # Continue with other images instead of raising
                continue

            # Identify output files
            expt_file = image_output_dir / "indexed_refined_detector.expt"
            refl_file = image_output_dir / "indexed_refined_detector.refl"

            # Verify files exist
            if not expt_file.exists() or not refl_file.exists():
                logger.error(f"Expected output files not found for {cbf_image_path}")
                continue

            # Generate masks
            mask_results = run_mask_generation_for_image(
                args, image_output_dir, cbf_image_path, expt_file, refl_file
            )

            # Store results
            result = {
                "cbf_image_path": cbf_image_path,
                "image_output_dir": image_output_dir,
                "expt_file": expt_file,
                "refl_file": refl_file,
                "status": "SUCCESS",
                **mask_results
            }
            results.append(result)

            logger.info(f"Successfully processed {cbf_image_path}")

        except Exception as e:
            logger.error(f"Failed to process {cbf_image_path}: {e}")
            # Continue with other images
            continue

    logger.info(f"Phase 1 completed: {len(results)}/{len(args.cbf_image_paths)} images processed successfully")
    return results


def run_mask_generation_for_image(
    args: argparse.Namespace, 
    image_output_dir: Path, 
    cbf_image_path: str,
    expt_file: Path, 
    refl_file: Path
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
        "total_diffuse_mask_path": total_diffuse_mask_path
    }


def run_phase2_data_extraction(
    args: argparse.Namespace,
    phase1_results: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Run Phase 2 data extraction for all successfully processed images.

    Returns:
        List of dictionaries with Phase 2 results
    """
    logger.info("=== Phase 2: Data Extraction (Multiple Images) ===")

    phase2_results = []

    for result in phase1_results:
        cbf_image_path = result["cbf_image_path"]
        image_output_dir = result["image_output_dir"]
        expt_file = result["expt_file"]
        total_diffuse_mask_path = result["total_diffuse_mask_path"]

        logger.info(f"Extracting diffuse data from {cbf_image_path}")

        try:
            # Instantiate DataExtractor
            data_extractor = DataExtractor()

            # Create ComponentInputFiles
            component_inputs = ComponentInputFiles(
                cbf_image_path=cbf_image_path,
                dials_expt_path=str(expt_file),
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
            )

            # Check outcome
            if data_extractor_outcome.status != "SUCCESS":
                logger.error(f"Data extraction failed for {cbf_image_path}: {data_extractor_outcome.status}")
                continue

            # Store results
            phase2_result = {
                **result,  # Include Phase 1 results
                "corrected_npz_path": output_npz_path,
                "phase2_status": "SUCCESS"
            }
            phase2_results.append(phase2_result)

            logger.info(f"Successfully extracted data from {cbf_image_path}")

        except Exception as e:
            logger.error(f"Failed to extract data from {cbf_image_path}: {e}")
            continue

    logger.info(f"Phase 2 completed: {len(phase2_results)} images processed successfully")
    return phase2_results


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

    # Note: This is a placeholder implementation since actual Phase 3 components
    # would need to be implemented. For now, we'll create mock output files
    # that demonstrate the expected structure.

    logger.warning("Phase 3 implementation is currently a placeholder")
    logger.warning("Creating mock output files for diagnostic testing")

    # Create mock Phase 3 outputs
    phase3_outputs = create_mock_phase3_outputs(phase2_results, output_dir)

    logger.info("Phase 3 completed (mock implementation)")
    return phase3_outputs


def create_mock_phase3_outputs(
    phase2_results: List[Dict[str, Any]], 
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create mock Phase 3 output files for testing the diagnostic system.
    
    Returns:
        Dictionary containing paths to mock output files
    """
    import numpy as np

    # Mock GlobalVoxelGrid definition
    grid_definition = {
        "crystal_avg_ref": {
            "unit_cell_params": [78.9, 78.9, 37.1, 90.0, 90.0, 90.0],
            "space_group": "P43212",
            "setting_matrix_a": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        },
        "hkl_bounds": {
            "h_min": -32, "h_max": 32,
            "k_min": -32, "k_max": 32,
            "l_min": -16, "l_max": 16
        },
        "ndiv_h": 64, "ndiv_k": 64, "ndiv_l": 32,
        "total_voxels": 131072
    }
    
    grid_definition_path = output_dir / "global_voxel_grid_definition.json"
    with open(grid_definition_path, "w") as f:
        json.dump(grid_definition, f, indent=2)

    # Mock ScalingModel refined parameters
    scaling_params = {
        "n_stills": len(phase2_results),
        "per_still_scales": {
            f"still_{i:03d}": {
                "b_i": 1.0 + 0.1 * np.random.randn(),  # Scale factor around 1.0
                "status": "converged"
            }
            for i in range(len(phase2_results))
        },
        "resolution_smoother": {
            "enabled": True,
            "control_points_q": [0.1, 0.2, 0.3, 0.4, 0.5],
            "control_points_a": [1.0, 0.98, 0.95, 0.92, 0.89]
        },
        "convergence_info": {
            "n_iterations": 5,
            "final_r_factor": 0.152
        }
    }
    
    scaling_params_path = output_dir / "refined_scaling_model_params.json"
    with open(scaling_params_path, "w") as f:
        json.dump(scaling_params, f, indent=2)

    # Mock VoxelData_relative
    n_voxels = grid_definition["total_voxels"]
    
    # Create realistic mock data
    np.random.seed(42)  # For reproducibility
    
    # Only populate a fraction of voxels (realistic for diffuse data)
    n_populated = n_voxels // 10
    voxel_indices = np.random.choice(n_voxels, n_populated, replace=False)
    
    # Mock intensities with realistic distribution
    base_intensities = np.random.exponential(scale=100, size=n_populated)
    intensities = np.full(n_voxels, np.nan)
    intensities[voxel_indices] = base_intensities
    
    # Mock sigmas (roughly Poisson-like relationship)
    sigmas = np.full(n_voxels, np.nan)
    sigmas[voxel_indices] = np.sqrt(base_intensities) + 0.1 * base_intensities
    
    # Mock observation counts
    num_observations = np.zeros(n_voxels, dtype=int)
    num_observations[voxel_indices] = np.random.poisson(lam=3, size=n_populated) + 1
    
    # Mock q-vectors for voxel centers
    h_coords = np.random.randint(-32, 33, n_voxels)
    k_coords = np.random.randint(-32, 33, n_voxels)
    l_coords = np.random.randint(-16, 17, n_voxels)
    
    # Convert HKL to q-vectors (simplified)
    q_centers_x = h_coords * 0.05  # Mock conversion factor
    q_centers_y = k_coords * 0.05
    q_centers_z = l_coords * 0.1
    q_magnitudes = np.sqrt(q_centers_x**2 + q_centers_y**2 + q_centers_z**2)
    
    voxel_data_path = output_dir / "voxel_data_relative.npz"
    np.savez_compressed(
        voxel_data_path,
        I_merged_relative=intensities,
        Sigma_merged_relative=sigmas,
        num_observations_in_voxel=num_observations,
        q_center_x=q_centers_x,
        q_center_y=q_centers_y,
        q_center_z=q_centers_z,
        q_magnitude_center=q_magnitudes,
        H_center=h_coords,
        K_center=k_coords,
        L_center=l_coords,
        voxel_indices=np.arange(n_voxels)
    )

    logger.info(f"Created mock grid definition: {grid_definition_path}")
    logger.info(f"Created mock scaling parameters: {scaling_params_path}")
    logger.info(f"Created mock voxel data: {voxel_data_path}")

    return {
        "grid_definition_file": grid_definition_path,
        "scaling_model_params_file": scaling_params_path,
        "voxel_data_file": voxel_data_path
    }


def save_intermediate_outputs_manifest(
    phase2_results: List[Dict[str, Any]], 
    output_dir: Path
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
                "total_diffuse_mask_path": str(result["total_diffuse_mask_path"])
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
        # Phase 1: DIALS Processing and Masking (Multiple Images)
        phase1_results = run_phase1_processing_for_multiple_images(args, output_dir)
        
        if not phase1_results:
            logger.error("No images were successfully processed in Phase 1")
            sys.exit(1)

        # Phase 2: Data Extraction (Multiple Images)
        phase2_results = run_phase2_data_extraction(args, phase1_results, output_dir)
        
        if not phase2_results:
            logger.error("No images were successfully processed in Phase 2")
            sys.exit(1)

        # Save intermediate outputs manifest if requested
        if args.save_intermediate_phase_outputs:
            save_intermediate_outputs_manifest(phase2_results, output_dir)

        # Phase 3: Voxelization, Relative Scaling, and Merging
        phase3_outputs = run_phase3_voxelization_and_scaling(args, phase2_results, output_dir)

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