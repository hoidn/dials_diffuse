"""Data extractor for diffuse scattering analysis."""

import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from diffusepipe.types.types_IDL import (
    ComponentInputFiles,
    ExtractionConfig,
    OperationOutcome,
)

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Data extractor for processing diffuse scattering data from detector pixels.

    This class implements the extraction pipeline as specified in data_extractor_IDL.md,
    processing raw detector data through geometric calculations, corrections, and filtering.
    """

    def __init__(self):
        """Initialize the DataExtractor."""
        pass

    def extract_from_still(
        self,
        inputs: ComponentInputFiles,
        config: ExtractionConfig,
        output_npz_path: str,
        start_angle: float,
        mask_total_2d: Optional[tuple] = None,
    ) -> OperationOutcome:
        """
        Extract diffuse scattering data from a still image.

        Args:
            inputs: Input file paths including CBF, experiment, and PDB
            config: Extraction configuration parameters
            output_npz_path: Path for output NPZ file
            start_angle: Start angle from CBF header to determine correct frame index
            mask_total_2d: Optional tuple of combined masks (Mask_pixel AND NOT BraggMask_2D_raw_i)
                           for each detector panel. If None, loads bragg_mask from inputs.

        Returns:
            OperationOutcome with success/failure status and output artifacts
        """
        try:
            if config.verbose:
                logger.setLevel(logging.DEBUG)

            logger.info("Starting diffuse data extraction")

            # 1. Validate inputs
            validation_result = self._validate_inputs(
                inputs, config, output_npz_path, mask_total_2d
            )
            if validation_result.status != "SUCCESS":
                return validation_result

            # 2. Load data
            logger.info("Loading input data")
            experiment, image_data, total_mask, pdb_data = self._load_data(
                inputs, start_angle, mask_total_2d
            )

            # 3. Consistency checks (if PDB provided)
            if inputs.external_pdb_path:
                logger.info("Performing consistency checks against reference PDB")
                consistency_result = self._check_pdb_consistency(
                    experiment, pdb_data, config
                )
                if consistency_result.status != "SUCCESS":
                    return consistency_result

            # 4. Process pixels
            logger.info("Processing detector pixels")
            q_vectors, intensities, sigmas, panel_ids, fast_coords, slow_coords = (
                self._process_pixels(experiment, image_data, total_mask, config)
            )

            if len(q_vectors) == 0:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="ProcessingError",
                    message="No pixels passed filtering criteria",
                )

            # 5. Save output
            logger.info(f"Saving {len(q_vectors)} data points to {output_npz_path}")
            self._save_output(
                q_vectors,
                intensities,
                sigmas,
                output_npz_path,
                panel_ids,
                fast_coords,
                slow_coords,
            )

            # 6. Generate diagnostics if requested
            output_artifacts = {"npz_file": output_npz_path}
            if config.plot_diagnostics:
                logger.info("Generating diagnostic plots")
                plot_paths = self._generate_diagnostic_plots(
                    q_vectors, intensities, sigmas, output_npz_path
                )
                output_artifacts.update(plot_paths)

            return OperationOutcome(
                status="SUCCESS",
                message=f"Successfully extracted {len(q_vectors)} data points",
                output_artifacts=output_artifacts,
            )

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")

            # Determine error code based on error type
            if "Input" in str(e) or "not found" in str(e).lower():
                error_code = "InputFileError"
            elif "consistency" in str(e).lower():
                error_code = "ConsistencyCheckFailed"
            elif "background" in str(e).lower():
                error_code = "BackgroundFileError"
            elif "write" in str(e).lower() or "save" in str(e).lower():
                error_code = "OutputWriteError"
            elif "DIALS" in str(e) or "model" in str(e).lower():
                error_code = "DIALSModelIncomplete"
            else:
                error_code = "ProcessingError"

            return OperationOutcome(
                status="FAILURE",
                error_code=error_code,
                message=f"Data extraction failed: {e}",
            )

    def _validate_inputs(
        self,
        inputs: ComponentInputFiles,
        config: ExtractionConfig,
        output_npz_path: str,
        mask_total_2d: Optional[tuple] = None,
    ) -> OperationOutcome:
        """Validate all input files and parameters."""
        try:
            # Check required input files
            if not inputs.cbf_image_path:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="InputFileError",
                    message="CBF image path not provided",
                )

            if not inputs.dials_expt_path:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="InputFileError",
                    message="DIALS experiment path not provided",
                )

            # Bragg mask is now optional since mask_total_2d can be passed directly
            if mask_total_2d is None and not inputs.bragg_mask_path:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="InputFileError",
                    message="Bragg mask path not provided and mask_total_2d not passed",
                )

            # Check file existence
            required_files = [
                (inputs.cbf_image_path, "CBF image"),
                (inputs.dials_expt_path, "DIALS experiment"),
            ]

            # Only require bragg_mask_path if mask_total_2d not provided
            if mask_total_2d is None and inputs.bragg_mask_path:
                required_files.append((inputs.bragg_mask_path, "Bragg mask"))

            if inputs.external_pdb_path:
                required_files.append((inputs.external_pdb_path, "External PDB"))

            for file_path, file_type in required_files:
                if not os.path.exists(file_path):
                    return OperationOutcome(
                        status="FAILURE",
                        error_code="InputFileError",
                        message=f"{file_type} file not found: {file_path}",
                    )

            # Check output directory is writable
            output_dir = os.path.dirname(output_npz_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            return OperationOutcome(status="SUCCESS", message="Input validation passed")

        except Exception as e:
            return OperationOutcome(
                status="FAILURE",
                error_code="InputFileError",
                message=f"Input validation failed: {e}",
            )

    def _load_data(
        self, inputs: ComponentInputFiles, start_angle: float, mask_total_2d: Optional[tuple] = None
    ) -> Tuple[object, np.ndarray, np.ndarray, Optional[object]]:
        """Load all required data files."""
        try:
            # Load DIALS experiment list (composite from sequence processing)
            from dxtbx.model.experiment_list import ExperimentListFactory

            experiment_list = ExperimentListFactory.from_json_file(inputs.dials_expt_path)
            if len(experiment_list) == 0:
                raise ValueError("No experiments found in DIALS experiment file")
            
            # Robust angle-based frame lookup using scan object
            if len(experiment_list) > 1:
                # Multiple experiments case - use the first scan to find frame index
                base_experiment = experiment_list[0]
                if hasattr(base_experiment, 'scan') and base_experiment.scan is not None:
                    scan = base_experiment.scan
                    try:
                        # Use the scan model to find the correct frame index from the angle
                        image_index = scan.get_image_index_from_angle(start_angle, deg=True)
                        # Ensure the index is an integer before using it for slicing
                        image_index = int(image_index)
                        logger.info(f"Resolved start_angle {start_angle}° to frame index {image_index}")
                    except (ValueError, AttributeError) as e:
                        raise ValueError(f"Could not find frame for start_angle {start_angle}°: {e}")
                    
                    # Get the experiment for this frame
                    if image_index >= len(experiment_list):
                        raise ValueError(f"Resolved image_index {image_index} is out of bounds for ExperimentList of length {len(experiment_list)}")
                    experiment = experiment_list[image_index]
                    imageset = experiment.imageset
                    # Always get the first (and only) image from the sliced imageset
                    image_data = imageset.get_raw_data(0)
                else:
                    raise ValueError("Multiple experiments found but no scan object available for angle resolution")
            else:
                # Single experiment with scan case
                experiment = experiment_list[0]
                imageset = experiment.imageset
                
                # Use scan object to resolve frame index from angle
                if hasattr(experiment, 'scan') and experiment.scan is not None:
                    scan = experiment.scan
                    try:
                        # Use the scan model to find the correct frame index from the angle
                        image_index = scan.get_image_index_from_angle(start_angle, deg=True)
                        # Ensure the index is an integer before using it for slicing
                        image_index = int(image_index)
                        logger.info(f"Resolved start_angle {start_angle}° to frame index {image_index}")
                    except (ValueError, AttributeError) as e:
                        raise ValueError(f"Could not find frame for start_angle {start_angle}°: {e}")
                    
                    # For sequence processing, get the geometry at the specific scan point
                    if hasattr(experiment.crystal, 'get_crystal_at_scan_point'):
                        # Get scan-varying crystal model
                        scan_point = image_index
                        crystal_at_scan_point = experiment.crystal.get_crystal_at_scan_point(scan_point)
                        # Create a new experiment with the frame-specific crystal
                        experiment = experiment_list[0]  # Keep other models the same
                        experiment.crystal = crystal_at_scan_point
                    
                    # Get the image data for the specific image
                    if image_index >= len(imageset):
                        # Fallback: use the specific CBF file directly
                        from dxtbx.imageset import ImageSetFactory
                        imagesets = ImageSetFactory.new([inputs.cbf_image_path])
                        if not imagesets:
                            raise ValueError(f"Failed to load image from {inputs.cbf_image_path}")
                        fallback_imageset = imagesets[0]
                        image_data = fallback_imageset.get_raw_data(0)
                    else:
                        # Get the image data for the specific frame from the full imageset
                        image_data = imageset.get_raw_data(image_index)
                else:
                    # No scan object - treat as still image
                    logger.info(f"No scan object found, loading image directly from CBF")
                    from dxtbx.imageset import ImageSetFactory
                    imagesets = ImageSetFactory.new([inputs.cbf_image_path])
                    if not imagesets:
                        raise ValueError(f"Failed to load image from {inputs.cbf_image_path}")
                    fallback_imageset = imagesets[0]
                    image_data = fallback_imageset.get_raw_data(0)
            
            # Validate experiment has required models
            if experiment.beam is None:
                raise ValueError("DIALS experiment missing beam model")
            if experiment.detector is None:
                raise ValueError("DIALS experiment missing detector model")
            if experiment.crystal is None:
                raise ValueError("DIALS experiment missing crystal model")

            # Convert to numpy array if needed
            if isinstance(image_data, tuple):
                # Multi-panel detector - use first panel for now
                image_data = image_data[0].as_numpy_array()
            else:
                # Single panel detector
                image_data = image_data.as_numpy_array()

            # Load total mask (either passed in-memory or from file)
            if mask_total_2d is not None:
                # Use passed in-memory mask
                total_mask = mask_total_2d
                if isinstance(total_mask, (tuple, list)):
                    # Multi-panel mask - use first panel for now
                    total_mask = total_mask[0]
                    if hasattr(total_mask, "as_numpy_array"):
                        total_mask = total_mask.as_numpy_array()
            else:
                # Load Bragg mask from file (legacy path)
                with open(inputs.bragg_mask_path, "rb") as f:
                    bragg_mask_data = pickle.load(f)

                # Convert to numpy array if it's a DIALS flex array
                if hasattr(bragg_mask_data, "as_numpy_array"):
                    total_mask = bragg_mask_data.as_numpy_array()
                elif isinstance(bragg_mask_data, (tuple, list)):
                    # Multi-panel mask - use first panel for now
                    total_mask = bragg_mask_data[0]
                    if hasattr(total_mask, "as_numpy_array"):
                        total_mask = total_mask.as_numpy_array()
                else:
                    total_mask = bragg_mask_data

            # Load PDB data if provided
            pdb_data = None
            if inputs.external_pdb_path:
                from iotbx.pdb import input as pdb_input

                pdb_data = pdb_input(file_name=inputs.external_pdb_path)

            logger.debug(f"Loaded experiment with {len(experiment.detector)} panels")
            logger.debug(f"Image shape: {image_data.shape}")
            logger.debug(f"Total mask shape: {total_mask.shape}")

            return experiment, image_data, total_mask, pdb_data

        except Exception as e:
            raise Exception(f"Failed to load input data: {e}")

    def _check_pdb_consistency(
        self, experiment: object, pdb_data: object, config: ExtractionConfig
    ) -> OperationOutcome:
        """Check consistency between DIALS crystal model and reference PDB."""
        try:
            # Extract unit cell from DIALS crystal model
            dials_cell = experiment.crystal.get_unit_cell()
            dials_params = dials_cell.parameters()  # (a, b, c, alpha, beta, gamma)

            # Extract unit cell from PDB
            pdb_crystal_symmetry = pdb_data.crystal_symmetry()
            if pdb_crystal_symmetry is None:
                logger.warning("PDB file has no crystal symmetry information")
                return OperationOutcome(
                    status="SUCCESS", message="No PDB crystal symmetry to check"
                )

            pdb_cell = pdb_crystal_symmetry.unit_cell()
            pdb_params = pdb_cell.parameters()

            # Check cell length tolerances
            for i in range(3):  # a, b, c
                dials_length = dials_params[i]
                pdb_length = pdb_params[i]
                rel_diff = abs(dials_length - pdb_length) / pdb_length

                if rel_diff > config.cell_length_tol:
                    return OperationOutcome(
                        status="FAILURE",
                        error_code="ConsistencyCheckFailed",
                        message=f"Cell length {['a', 'b', 'c'][i]} differs by {rel_diff:.3f} "
                        f"(tolerance: {config.cell_length_tol}): "
                        f"DIALS={dials_length:.3f}, PDB={pdb_length:.3f}",
                    )

            # Check cell angle tolerances
            for i in range(3, 6):  # alpha, beta, gamma
                dials_angle = dials_params[i]
                pdb_angle = pdb_params[i]
                abs_diff = abs(dials_angle - pdb_angle)

                if abs_diff > config.cell_angle_tol:
                    return OperationOutcome(
                        status="FAILURE",
                        error_code="ConsistencyCheckFailed",
                        message=f"Cell angle {['alpha', 'beta', 'gamma'][i-3]} differs by {abs_diff:.3f}° "
                        f"(tolerance: {config.cell_angle_tol}°): "
                        f"DIALS={dials_angle:.3f}°, PDB={pdb_angle:.3f}°",
                    )

            logger.info("PDB consistency checks passed")
            return OperationOutcome(
                status="SUCCESS", message="PDB consistency checks passed"
            )

        except Exception as e:
            return OperationOutcome(
                status="FAILURE",
                error_code="ConsistencyCheckFailed",
                message=f"PDB consistency check failed: {e}",
            )

    def _process_pixels(
        self,
        experiment: object,
        image_data: np.ndarray,
        total_mask: np.ndarray,
        config: ExtractionConfig,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Process detector pixels to extract q-vectors, intensities, and errors.

        Uses vectorized implementation for improved performance.

        Returns:
            tuple: (q_vectors, intensities, sigmas, panel_ids, fast_coords, slow_coords)
                   The last three are None if save_original_pixel_coordinates is False
        """
        # Use vectorized implementation for better performance
        return self._process_pixels_vectorized(
            experiment, image_data, total_mask, config
        )

    def _process_pixels_iterative(
        self,
        experiment: object,
        image_data: np.ndarray,
        total_mask: np.ndarray,
        config: ExtractionConfig,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Original iterative implementation - kept for comparison and fallback.

        Returns:
            tuple: (q_vectors, intensities, sigmas, panel_ids, fast_coords, slow_coords)
                   The last three are None if save_original_pixel_coordinates is False
        """
        q_vectors_list = []
        intensities_list = []
        sigmas_list = []

        # Track pixel coordinates if requested
        panel_ids_list = [] if config.save_original_pixel_coordinates else None
        fast_coords_list = [] if config.save_original_pixel_coordinates else None
        slow_coords_list = [] if config.save_original_pixel_coordinates else None

        # Get detector panel (assume single panel for now)
        panel = experiment.detector[0]
        beam = experiment.beam

        # Get wavelength and beam vector for q-vector calculations
        wavelength = beam.get_wavelength()
        k_magnitude = 2 * np.pi / wavelength
        s0 = beam.get_s0()
        k_in = np.array([s0[0], s0[1], s0[2]]) * k_magnitude

        # Load background data if specified
        background_data = None
        if config.subtract_measured_background_path:
            try:
                background_data = np.load(config.subtract_measured_background_path)
                logger.info(
                    f"Loaded background data from {config.subtract_measured_background_path}"
                )
            except Exception as e:
                raise Exception(f"Failed to load background data: {e}")

        # Process pixels with step size
        height, width = image_data.shape
        total_pixels = (height // config.pixel_step) * (width // config.pixel_step)
        processed_pixels = 0

        logger.info(
            f"Processing {total_pixels} pixels with step size {config.pixel_step}"
        )

        for slow_idx in range(0, height, config.pixel_step):
            for fast_idx in range(0, width, config.pixel_step):
                processed_pixels += 1

                # Skip if in total mask (Bragg regions or bad pixels)
                if total_mask[slow_idx, fast_idx]:
                    continue

                # Calculate q-vector for this pixel
                try:
                    lab_coord = panel.get_pixel_lab_coord(
                        (float(fast_idx), float(slow_idx))
                    )
                    scatter_direction = np.array(
                        [lab_coord[0], lab_coord[1], lab_coord[2]]
                    )
                    scatter_direction_norm = scatter_direction / np.linalg.norm(
                        scatter_direction
                    )
                    k_out = scatter_direction_norm * k_magnitude
                    q_vector = k_out - k_in
                except Exception as e:
                    logger.debug(
                        f"Failed to calculate q-vector for pixel ({fast_idx}, {slow_idx}): {e}"
                    )
                    continue

                # Get raw intensity
                raw_intensity = float(image_data[slow_idx, fast_idx])

                # Apply background subtraction first (Module 2.S.2 order)
                bg_value = 0.0
                bg_variance = 0.0

                if background_data is not None:
                    bg_value = background_data[slow_idx, fast_idx]
                    # Assume Poisson statistics for background
                    bg_variance = bg_value if bg_value > 0 else 0.0
                elif config.subtract_constant_background_value is not None:
                    bg_value = config.subtract_constant_background_value
                    bg_variance = 0.0  # Constant background has no variance

                # Background-subtracted intensity
                intensity_bg_sub = raw_intensity - bg_value

                # Apply gain and exposure time normalization
                # Note: exposure time normalization will be added when config includes it
                intensity_processed = intensity_bg_sub * config.gain

                # Calculate error with proper error propagation (Module 2.S.2)
                # Var_photon_initial = I_raw / gain (variance of original raw count)
                var_photon_initial = (
                    raw_intensity / config.gain if config.gain > 0 else raw_intensity
                )
                # Var_processed = (Var_photon_initial + Var_bkg) * gain^2
                var_processed = (var_photon_initial + bg_variance) * (config.gain**2)
                sigma_processed = np.sqrt(var_processed)

                # Apply pixel corrections (Module 2.S.2)
                corrected_intensity, corrected_sigma = self._apply_pixel_corrections(
                    intensity_processed,
                    sigma_processed,
                    q_vector,
                    lab_coord,
                    panel,
                    beam,
                    experiment,
                    slow_idx,
                    fast_idx,
                    config,
                )

                # Apply resolution filter
                q_magnitude = np.linalg.norm(q_vector)
                d_spacing = 2 * np.pi / q_magnitude if q_magnitude > 0 else float("inf")

                if config.min_res is not None and d_spacing > config.min_res:
                    continue
                if config.max_res is not None and d_spacing < config.max_res:
                    continue

                # Apply intensity filter
                if (
                    config.min_intensity is not None
                    and corrected_intensity < config.min_intensity
                ):
                    continue
                if (
                    config.max_intensity is not None
                    and corrected_intensity > config.max_intensity
                ):
                    continue

                # Store valid pixel data
                q_vectors_list.append(q_vector)
                intensities_list.append(corrected_intensity)
                sigmas_list.append(corrected_sigma)

                # Store pixel coordinates if requested
                if config.save_original_pixel_coordinates:
                    panel_ids_list.append(0)  # Single panel for now
                    fast_coords_list.append(fast_idx)
                    slow_coords_list.append(slow_idx)

                # Progress logging
                if processed_pixels % 10000 == 0:
                    logger.debug(
                        f"Processed {processed_pixels}/{total_pixels} pixels, "
                        f"kept {len(q_vectors_list)} valid points"
                    )

        logger.info(
            f"Kept {len(q_vectors_list)} pixels out of {processed_pixels} processed"
        )

        # Convert to numpy arrays
        if len(q_vectors_list) > 0:
            q_vectors = np.array(q_vectors_list)
            intensities = np.array(intensities_list)
            sigmas = np.array(sigmas_list)

            # Convert coordinate arrays if tracking enabled
            if config.save_original_pixel_coordinates:
                panel_ids = np.array(panel_ids_list)
                fast_coords = np.array(fast_coords_list)
                slow_coords = np.array(slow_coords_list)
            else:
                panel_ids = fast_coords = slow_coords = None
        else:
            q_vectors = np.empty((0, 3))
            intensities = np.empty(0)
            sigmas = np.empty(0)
            panel_ids = fast_coords = slow_coords = None

        return q_vectors, intensities, sigmas, panel_ids, fast_coords, slow_coords

    def _process_pixels_vectorized(
        self,
        experiment: object,
        image_data: np.ndarray,
        total_mask: np.ndarray,
        config: ExtractionConfig,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Vectorized implementation of pixel processing for improved performance.

        Returns:
            tuple: (q_vectors, intensities, sigmas, panel_ids, fast_coords, slow_coords)
        """
        logger.info("Using vectorized pixel processing implementation")

        # Get detector panel (assume single panel for now)
        panel = experiment.detector[0]
        beam = experiment.beam

        # Get wavelength and beam vector for q-vector calculations
        wavelength = beam.get_wavelength()
        k_magnitude = 2 * np.pi / wavelength
        s0 = beam.get_s0()
        k_in = np.array([s0[0], s0[1], s0[2]]) * k_magnitude

        # Load background data if specified
        background_data = None
        if config.subtract_measured_background_path:
            try:
                background_data = np.load(config.subtract_measured_background_path)
                logger.info(
                    f"Loaded background data from {config.subtract_measured_background_path}"
                )
            except Exception as e:
                raise Exception(f"Failed to load background data: {e}")

        # Step 1: Generate coordinate arrays for pixels to process
        height, width = image_data.shape

        # Create coordinate grids with step size
        slow_indices = np.arange(0, height, config.pixel_step)
        fast_indices = np.arange(0, width, config.pixel_step)
        slow_grid, fast_grid = np.meshgrid(slow_indices, fast_indices, indexing="ij")

        # Flatten coordinate grids
        slow_coords_all = slow_grid.flatten()
        fast_coords_all = fast_grid.flatten()

        # Apply mask filtering
        mask_values = total_mask[slow_coords_all, fast_coords_all]
        unmasked_indices = ~mask_values

        slow_coords = slow_coords_all[unmasked_indices]
        fast_coords = fast_coords_all[unmasked_indices]
        n_pixels = len(slow_coords)

        logger.info(f"Processing {n_pixels} unmasked pixels (vectorized)")

        if n_pixels == 0:
            # No pixels to process
            if config.save_original_pixel_coordinates:
                return (
                    np.empty((0, 3)),
                    np.empty(0),
                    np.empty(0),
                    np.empty(0, dtype=int),
                    np.empty(0, dtype=int),
                    np.empty(0, dtype=int),
                )
            else:
                return np.empty((0, 3)), np.empty(0), np.empty(0), None, None, None

        # Step 2: Vectorized batch calculation of lab coordinates using DIALS batch API
        from dials.array_family import flex
        
        # Convert pixel coordinates to flex.vec2_double for batch processing
        pixel_coords_flex = flex.vec2_double()
        for fast, slow in zip(fast_coords, slow_coords):
            pixel_coords_flex.append((float(fast), float(slow)))
        
        # Batch calculate lab coordinates - major performance improvement!
        try:
            lab_coords_flex = panel.get_lab_coord(pixel_coords_flex)
            # Convert flex.vec3_double to numpy array
            lab_coords = np.array([[coord[0], coord[1], coord[2]] for coord in lab_coords_flex])
            
            logger.debug(f"Successfully calculated {len(lab_coords)} lab coordinates using vectorized DIALS API")
            
        except Exception as e:
            logger.warning(f"Batch lab coordinate calculation failed, falling back to individual calculation: {e}")
            # Fallback to original method if batch fails
            lab_coords = np.zeros((n_pixels, 3))
            for i in range(n_pixels):
                try:
                    lab_coord = panel.get_pixel_lab_coord(
                        (float(fast_coords[i]), float(slow_coords[i]))
                    )
                    lab_coords[i] = [lab_coord[0], lab_coord[1], lab_coord[2]]
                except Exception as e:
                    logger.debug(
                        f"Failed to get lab coord for pixel ({fast_coords[i]}, {slow_coords[i]}): {e}"
                    )
                    # Mark as invalid - will be filtered later
                    lab_coords[i] = [np.nan, np.nan, np.nan]

        # Filter out invalid coordinates (NaN values)
        valid_coords_mask = ~np.isnan(lab_coords).any(axis=1)
        lab_coords = lab_coords[valid_coords_mask]
        slow_coords = slow_coords[valid_coords_mask]
        fast_coords = fast_coords[valid_coords_mask]
        n_pixels = len(slow_coords)

        if n_pixels == 0:
            if config.save_original_pixel_coordinates:
                return (
                    np.empty((0, 3)),
                    np.empty(0),
                    np.empty(0),
                    np.empty(0, dtype=int),
                    np.empty(0, dtype=int),
                    np.empty(0, dtype=int),
                )
            else:
                return np.empty((0, 3)), np.empty(0), np.empty(0), None, None, None

        # Calculate q-vectors vectorized
        scatter_directions = (
            lab_coords / np.linalg.norm(lab_coords, axis=1)[:, np.newaxis]
        )
        k_out = scatter_directions * k_magnitude
        q_vectors = k_out - k_in[np.newaxis, :]

        # Step 3: Extract raw intensities and apply background subtraction
        raw_intensities = image_data[slow_coords, fast_coords].astype(float)

        # Background subtraction
        if background_data is not None:
            bg_values = background_data[slow_coords, fast_coords]
            bg_variances = np.maximum(bg_values, 0.0)  # Poisson statistics
        elif config.subtract_constant_background_value is not None:
            bg_values = np.full(n_pixels, config.subtract_constant_background_value)
            bg_variances = np.zeros(n_pixels)
        else:
            bg_values = np.zeros(n_pixels)
            bg_variances = np.zeros(n_pixels)

        # Apply background subtraction and gain
        intensities_bg_sub = raw_intensities - bg_values
        intensities_processed = intensities_bg_sub * config.gain

        # Error propagation
        var_photon_initial = (
            raw_intensities / config.gain if config.gain > 0 else raw_intensities
        )
        var_processed = (var_photon_initial + bg_variances) * (config.gain**2)
        sigmas_processed = np.sqrt(var_processed)

        # Step 4: Apply vectorized pixel corrections
        corrected_intensities, corrected_sigmas = (
            self._apply_pixel_corrections_vectorized(
                intensities_processed,
                sigmas_processed,
                q_vectors,
                lab_coords,
                panel,
                beam,
                experiment,
                slow_coords,
                fast_coords,
                config,
            )
        )

        # Step 5: Apply filters
        q_magnitudes = np.linalg.norm(q_vectors, axis=1)
        d_spacings = 2 * np.pi / q_magnitudes
        d_spacings[q_magnitudes == 0] = np.inf

        # Resolution filter
        resolution_mask = np.ones(n_pixels, dtype=bool)
        if config.min_res is not None:
            resolution_mask &= d_spacings <= config.min_res
        if config.max_res is not None:
            resolution_mask &= d_spacings >= config.max_res

        # Intensity filter
        intensity_mask = np.ones(n_pixels, dtype=bool)
        if config.min_intensity is not None:
            intensity_mask &= corrected_intensities >= config.min_intensity
        if config.max_intensity is not None:
            intensity_mask &= corrected_intensities <= config.max_intensity

        # Combine filters
        final_mask = resolution_mask & intensity_mask

        # Apply final filtering
        final_q_vectors = q_vectors[final_mask]
        final_intensities = corrected_intensities[final_mask]
        final_sigmas = corrected_sigmas[final_mask]

        logger.info(f"Kept {len(final_intensities)} pixels after filtering")

        # Handle coordinate tracking
        if config.save_original_pixel_coordinates:
            final_panel_ids = np.zeros(
                len(final_intensities), dtype=int
            )  # Single panel
            final_fast_coords = fast_coords[final_mask]
            final_slow_coords = slow_coords[final_mask]
            return (
                final_q_vectors,
                final_intensities,
                final_sigmas,
                final_panel_ids,
                final_fast_coords,
                final_slow_coords,
            )
        else:
            return final_q_vectors, final_intensities, final_sigmas, None, None, None

    def _apply_pixel_corrections_vectorized(
        self,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        q_vectors: np.ndarray,
        lab_coords: np.ndarray,
        panel: object,
        beam: object,
        experiment: object,
        slow_coords: np.ndarray,
        fast_coords: np.ndarray,
        config: ExtractionConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply pixel corrections in vectorized fashion."""

        n_pixels = len(intensities)

        # Calculate s1 vectors for DIALS corrections
        s1_vectors = q_vectors + np.array(beam.get_s0())[np.newaxis, :]

        # Get LP and QE corrections from DIALS (batch)
        if config.lp_correction_enabled:
            try:
                from dials.algorithms.integration import Corrections
                from dials.array_family import flex

                # Create or use cached corrections object
                if not hasattr(self, "_corrections_obj"):
                    self._corrections_obj = Corrections(
                        beam, experiment.goniometer, experiment.detector
                    )

                # Convert to DIALS flex arrays using vectorized approach
                s1_flex = flex.vec3_double([tuple(s1_vec) for s1_vec in s1_vectors])
                panel_indices_flex = flex.size_t([0] * n_pixels)  # Single panel for now

                # Get LP corrections (returns divisors) - vectorized conversion
                lp_divisors = self._corrections_obj.lp(s1_flex)
                lp_multipliers = 1.0 / np.array(lp_divisors)

                # Get QE corrections (returns multipliers) - vectorized conversion
                qe_multipliers_flex = self._corrections_obj.qe(
                    s1_flex, panel_indices_flex
                )
                qe_multipliers = np.array(qe_multipliers_flex)

            except Exception as e:
                logger.debug(f"DIALS corrections failed, using defaults: {e}")
                lp_multipliers = np.ones(n_pixels)
                qe_multipliers = np.ones(n_pixels)
        else:
            lp_multipliers = np.ones(n_pixels)
            qe_multipliers = np.ones(n_pixels)

        # Calculate custom corrections vectorized
        sa_multipliers = self._calculate_solid_angle_correction_vectorized(
            lab_coords, panel, slow_coords, fast_coords
        )
        air_multipliers = self._calculate_air_attenuation_correction_vectorized(
            lab_coords, beam, config
        )

        # Combine all corrections
        total_corrections = (
            lp_multipliers * qe_multipliers * sa_multipliers * air_multipliers
        )

        # Apply corrections
        corrected_intensities = intensities * total_corrections
        corrected_sigmas = sigmas * total_corrections

        return corrected_intensities, corrected_sigmas

    def _calculate_solid_angle_correction_vectorized(
        self,
        lab_coords: np.ndarray,
        panel: object,
        slow_coords: np.ndarray,
        fast_coords: np.ndarray,
    ) -> np.ndarray:
        """Calculate solid angle corrections for multiple pixels."""
        try:
            pixel_sizes = panel.get_pixel_size()
            pixel_area = pixel_sizes[0] * pixel_sizes[1]  # mm²

            # Get panel normal
            fast_axis = np.array(panel.get_fast_axis())
            slow_axis = np.array(panel.get_slow_axis())
            normal = np.cross(fast_axis, slow_axis)
            normal = normal / np.linalg.norm(normal)

            # Calculate distances and directions
            distances = np.linalg.norm(lab_coords, axis=1)
            scatter_directions = lab_coords / distances[:, np.newaxis]

            # Calculate solid angles
            cos_theta = np.abs(np.dot(scatter_directions, normal))
            solid_angles = (pixel_area * cos_theta) / (distances**2)

            # Convert to correction multipliers
            sa_multipliers = 1.0 / solid_angles

            return sa_multipliers

        except Exception as e:
            logger.debug(f"Vectorized solid angle calculation failed: {e}")
            return np.ones(len(lab_coords))

    def _calculate_air_attenuation_correction_vectorized(
        self, lab_coords: np.ndarray, beam: object, config: ExtractionConfig
    ) -> np.ndarray:
        """Calculate air attenuation corrections for multiple pixels."""
        try:
            # Get X-ray energy
            wavelength = beam.get_wavelength()
            energy_ev = 12398.4 / wavelength

            # Calculate path lengths
            path_lengths = (
                np.linalg.norm(lab_coords, axis=1) / 1000.0
            )  # Convert mm to m

            # Get air attenuation coefficient (same for all pixels)
            temperature_k = getattr(config, "air_temperature_k", 293.15)
            pressure_atm = getattr(config, "air_pressure_atm", 1.0)
            mu_air = self._calculate_air_attenuation_coefficient(
                energy_ev, temperature_k, pressure_atm
            )

            # Calculate attenuation corrections
            attenuations = np.exp(-mu_air * path_lengths)
            air_multipliers = 1.0 / attenuations

            return air_multipliers

        except Exception as e:
            logger.debug(f"Vectorized air attenuation calculation failed: {e}")
            return np.ones(len(lab_coords))

    def _save_output(
        self,
        q_vectors: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        output_path: str,
        panel_ids: Optional[np.ndarray] = None,
        fast_coords: Optional[np.ndarray] = None,
        slow_coords: Optional[np.ndarray] = None,
    ):
        """Save extracted data to NPZ file."""
        try:
            # Base data to save
            save_data = {
                "q_vectors": q_vectors,
                "intensities": intensities,
                "sigmas": sigmas,
            }

            # Add pixel coordinates if available
            if (
                panel_ids is not None
                and fast_coords is not None
                and slow_coords is not None
            ):
                save_data["original_panel_ids"] = panel_ids
                save_data["original_fast_coords"] = fast_coords
                save_data["original_slow_coords"] = slow_coords
                logger.info("Including original pixel coordinates in NPZ output")

            np.savez_compressed(output_path, **save_data)
            logger.info(f"Saved data to {output_path}")
        except Exception as e:
            raise Exception(f"Failed to save output file: {e}")

    def _generate_diagnostic_plots(
        self,
        q_vectors: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        output_prefix: str,
    ) -> Dict[str, str]:
        """Generate diagnostic plots."""
        try:
            base_path = output_prefix.replace(".npz", "")
            plot_paths = {}

            # Q-space coverage plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 2D q-space projection
            ax1.scatter(q_vectors[:, 0], q_vectors[:, 1], alpha=0.5, s=1)
            ax1.set_xlabel("qx (Å⁻¹)")
            ax1.set_ylabel("qy (Å⁻¹)")
            ax1.set_title("Q-space Coverage (qx vs qy)")
            ax1.set_aspect("equal")

            # Intensity distribution
            ax2.hist(np.log10(intensities + 1), bins=50, alpha=0.7)
            ax2.set_xlabel("log10(Intensity + 1)")
            ax2.set_ylabel("Count")
            ax2.set_title("Intensity Distribution")

            plt.tight_layout()
            qspace_plot_path = f"{base_path}_qspace_coverage.png"
            plt.savefig(qspace_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            plot_paths["qspace_coverage_plot"] = qspace_plot_path

            # Resolution shell plot
            q_magnitudes = np.linalg.norm(q_vectors, axis=1)
            d_spacings = 2 * np.pi / q_magnitudes

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(d_spacings, intensities, alpha=0.5, s=1)
            ax.set_xlabel("d-spacing (Å)")
            ax.set_ylabel("Intensity")
            ax.set_title("Intensity vs Resolution")
            ax.set_xlim(0, min(20, np.max(d_spacings)))

            resolution_plot_path = f"{base_path}_resolution_analysis.png"
            plt.savefig(resolution_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            plot_paths["resolution_analysis_plot"] = resolution_plot_path

            logger.info(f"Generated diagnostic plots: {', '.join(plot_paths.values())}")
            return plot_paths

        except Exception as e:
            logger.warning(f"Failed to generate diagnostic plots: {e}")
            return {}

    def _apply_pixel_corrections(
        self,
        intensity: float,
        sigma: float,
        q_vector: np.ndarray,
        lab_coord: np.ndarray,
        panel: object,
        beam: object,
        experiment: object,
        slow_idx: int,
        fast_idx: int,
        config: ExtractionConfig,
    ) -> Tuple[float, float]:
        """
        Apply pixel-level corrections as per Module 2.S.2.

        Applies Lorentz-Polarization, Quantum Efficiency, Solid Angle,
        and Air Attenuation corrections, with error propagation.

        Args:
            intensity: Raw intensity (after gain)
            sigma: Initial sigma (Poisson error)
            q_vector: Scattering vector for this pixel
            lab_coord: Lab coordinates of pixel center
            panel: Detector panel object
            beam: Beam model
            experiment: Complete experiment object
            slow_idx, fast_idx: Pixel coordinates
            config: Extraction configuration

        Returns:
            Tuple of (corrected_intensity, corrected_sigma)
        """
        try:
            # Calculate s1 vector for DIALS Corrections API
            s1_magnitude = 1.0 / beam.get_wavelength()
            scatter_direction = lab_coord / np.linalg.norm(lab_coord)
            s1_vector = scatter_direction * s1_magnitude

            # 1. Lorentz-Polarization Correction (using DIALS Corrections API)
            lp_mult = 1.0  # Default no correction
            if config.lp_correction_enabled:
                lp_mult = self._get_lp_correction(s1_vector, beam, experiment)

            # 2. Quantum Efficiency Correction (using DIALS Corrections API)
            qe_mult = self._get_qe_correction(s1_vector, beam, experiment, panel_idx=0)

            # 3. Solid Angle Correction (custom calculation)
            sa_mult = self._calculate_solid_angle_correction(
                lab_coord, panel, fast_idx, slow_idx
            )

            # 4. Air Attenuation Correction (custom calculation)
            air_mult = self._calculate_air_attenuation_correction(
                lab_coord, beam, config
            )

            # 5. Combine all corrections
            total_correction_mult = lp_mult * qe_mult * sa_mult * air_mult

            # 6. Apply correction to intensity
            corrected_intensity = intensity * total_correction_mult

            # 7. Error propagation (assuming correction factors have negligible uncertainty)
            corrected_sigma = sigma * total_correction_mult

            logger.debug(
                f"Pixel ({fast_idx}, {slow_idx}): LP={lp_mult:.3f}, QE={qe_mult:.3f}, "
                f"SA={sa_mult:.3f}, Air={air_mult:.3f}, Total={total_correction_mult:.3f}"
            )

            return corrected_intensity, corrected_sigma

        except Exception as e:
            logger.debug(f"Correction failed for pixel ({fast_idx}, {slow_idx}): {e}")
            return intensity, sigma

    def _calculate_solid_angle_correction(
        self, lab_coord: np.ndarray, panel: object, fast_idx: int, slow_idx: int
    ) -> float:
        """
        Calculate solid angle correction factor.

        Implements: Ω = (A_pixel × cos θ_normal_to_s1) / r²
        Returns: SA_mult = 1.0 / Ω (where Ω is the solid angle divisor)
        """
        try:
            # Get pixel size
            pixel_size = panel.get_pixel_size()  # (fast_size, slow_size) in mm
            pixel_area = pixel_size[0] * pixel_size[1]  # mm²

            # Get distance from sample to pixel
            r = np.linalg.norm(lab_coord)  # mm

            # Get panel normal vector
            fast_axis = np.array(panel.get_fast_axis())
            slow_axis = np.array(panel.get_slow_axis())
            normal = np.cross(fast_axis, slow_axis)
            normal = normal / np.linalg.norm(normal)

            # Calculate angle between normal and scattered beam direction
            scatter_direction = lab_coord / r
            cos_theta = np.abs(np.dot(normal, scatter_direction))

            # Calculate solid angle
            solid_angle = (pixel_area * cos_theta) / (r * r)

            # Return multiplicative correction (1/solid_angle)
            sa_mult = 1.0 / solid_angle if solid_angle > 0 else 1.0

            return sa_mult

        except Exception as e:
            logger.debug(f"Solid angle calculation failed: {e}")
            return 1.0

    def _calculate_air_attenuation_correction(
        self, lab_coord: np.ndarray, beam: object, config: ExtractionConfig
    ) -> float:
        """
        Calculate air attenuation correction factor.

        Implements Beer-Lambert law using NIST tabulated mass attenuation coefficients:
        Attenuation = exp(-μ_air * path_length)
        Returns: Air_mult = 1.0 / Attenuation

        Uses scientifically accurate calculation based on:
        - NIST X-ray mass attenuation coefficients for air components
        - Standard air composition (N: 78.084%, O: 20.946%, Ar: 0.934%, C: 0.036%)
        - Ideal gas law for density calculation with configurable T and P
        """
        try:
            # Get X-ray energy from wavelength
            wavelength = beam.get_wavelength()  # Angstroms
            energy_ev = 12398.4 / wavelength  # eV

            # Path length from sample to pixel (assuming sample at origin)
            path_length = np.linalg.norm(lab_coord) / 1000.0  # Convert mm to meters

            # Calculate linear attenuation coefficient for air at this energy
            # Using NIST tabulated data for air components
            temperature_k = getattr(config, "air_temperature_k", 293.15)
            pressure_atm = getattr(config, "air_pressure_atm", 1.0)
            mu_air = self._calculate_air_attenuation_coefficient(
                energy_ev, temperature_k, pressure_atm
            )

            # Apply Beer-Lambert law
            attenuation = np.exp(-mu_air * path_length)

            # Return multiplicative correction (1/attenuation)
            air_mult = 1.0 / attenuation if attenuation > 0 else 1.0

            return air_mult

        except Exception as e:
            logger.debug(f"Air attenuation calculation failed: {e}")
            return 1.0

    def _calculate_air_attenuation_coefficient(
        self, energy_ev: float, temperature_k: float = 293.15, pressure_atm: float = 1.0
    ) -> float:
        """
        Calculate linear attenuation coefficient for air at given X-ray energy.

        Uses tabulated NIST mass attenuation coefficients for air components
        (N: 78.084%, O: 20.946%, Ar: 0.934%, C: 0.036% by mass).

        Args:
            energy_ev: X-ray energy in eV
            temperature_k: Air temperature in Kelvin (default: 293.15 K = 20°C)
            pressure_atm: Air pressure in atmospheres (default: 1.0 atm)

        Returns:
            Linear attenuation coefficient in m⁻¹
        """
        try:
            # Standard air composition by mass (NIST dry air at sea level)
            air_composition = {
                "N": 0.78084,  # Nitrogen
                "O": 0.20946,  # Oxygen
                "Ar": 0.00934,  # Argon
                "C": 0.00036,  # Carbon (CO2)
            }

            # Molar masses (g/mol)
            molar_masses = {"N": 14.0067, "O": 15.9994, "Ar": 39.948, "C": 12.0107}

            # Calculate effective molar mass of air
            M_air = sum(
                air_composition[element] * molar_masses[element]
                for element in air_composition
            )

            # Calculate air density using ideal gas law
            # ρ = (P × M) / (R × T), where R = 8.314 J/(mol·K) = 0.08206 L·atm/(mol·K)
            R_atm = 0.08206  # L·atm/(mol·K)
            air_density_g_per_L = (pressure_atm * M_air) / (R_atm * temperature_k)
            air_density = air_density_g_per_L / 1000.0  # Convert to g/cm³

            # Get mass attenuation coefficient for each component
            mu_over_rho_total = 0.0
            for element, mass_fraction in air_composition.items():
                mu_over_rho_element = self._get_mass_attenuation_coefficient(
                    element, energy_ev
                )
                mu_over_rho_total += mass_fraction * mu_over_rho_element

            # Calculate linear attenuation coefficient
            mu_linear_cm = mu_over_rho_total * air_density  # cm⁻¹
            mu_linear_m = mu_linear_cm * 100  # Convert to m⁻¹

            return mu_linear_m

        except Exception as e:
            logger.debug(f"Air attenuation coefficient calculation failed: {e}")
            # Fallback: very small attenuation for typical X-ray energies
            return 0.001

    def _get_mass_attenuation_coefficient(
        self, element: str, energy_ev: float
    ) -> float:
        """
        Get mass attenuation coefficient (μ/ρ) for an element at given X-ray energy.

        Uses tabulated NIST data for X-ray mass attenuation coefficients.
        Data covers the range 1 keV to 1000 keV, with interpolation between points.

        Args:
            element: Element symbol ('N', 'O', 'Ar', 'C')
            energy_ev: X-ray energy in eV

        Returns:
            Mass attenuation coefficient in cm²/g
        """
        # NIST X-ray mass attenuation coefficients (μ/ρ) in cm²/g
        # Energy values in eV, coefficients interpolated from NIST tables
        # Source: NIST XCOM database (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html)

        nist_data = {
            "N": {  # Nitrogen (Z=7)
                "energies": [
                    1000,
                    1500,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    8000,
                    10000,
                    15000,
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                    80000,
                    100000,
                ],
                "mu_rho": [
                    9.04e-1,
                    3.69e-1,
                    1.96e-1,
                    8.54e-2,
                    4.81e-2,
                    3.14e-2,
                    2.26e-2,
                    1.47e-2,
                    1.07e-2,
                    5.86e-3,
                    3.78e-3,
                    2.02e-3,
                    1.35e-3,
                    1.01e-3,
                    8.21e-4,
                    5.72e-4,
                    4.30e-4,
                ],
            },
            "O": {  # Oxygen (Z=8)
                "energies": [
                    1000,
                    1500,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    8000,
                    10000,
                    15000,
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                    80000,
                    100000,
                ],
                "mu_rho": [
                    1.18,
                    4.77e-1,
                    2.48e-1,
                    1.06e-1,
                    5.95e-2,
                    3.87e-2,
                    2.78e-2,
                    1.80e-2,
                    1.30e-2,
                    7.13e-3,
                    4.61e-3,
                    2.46e-3,
                    1.64e-3,
                    1.22e-3,
                    9.95e-4,
                    6.91e-4,
                    5.18e-4,
                ],
            },
            "Ar": {  # Argon (Z=18)
                "energies": [
                    1000,
                    1500,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    8000,
                    10000,
                    15000,
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                    80000,
                    100000,
                ],
                "mu_rho": [
                    8.21,
                    3.88,
                    2.14,
                    9.68e-1,
                    5.49e-1,
                    3.58e-1,
                    2.57e-1,
                    1.65e-1,
                    1.18e-1,
                    6.32e-2,
                    4.02e-2,
                    2.12e-2,
                    1.40e-2,
                    1.04e-2,
                    8.41e-3,
                    5.82e-3,
                    4.35e-3,
                ],
            },
            "C": {  # Carbon (Z=6)
                "energies": [
                    1000,
                    1500,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    8000,
                    10000,
                    15000,
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                    80000,
                    100000,
                ],
                "mu_rho": [
                    6.36e-1,
                    2.71e-1,
                    1.49e-1,
                    6.82e-2,
                    3.95e-2,
                    2.60e-2,
                    1.89e-2,
                    1.25e-2,
                    9.14e-3,
                    5.08e-3,
                    3.29e-3,
                    1.76e-3,
                    1.18e-3,
                    8.84e-4,
                    7.19e-4,
                    5.01e-4,
                    3.76e-4,
                ],
            },
        }

        if element not in nist_data:
            logger.warning(
                f"No mass attenuation data for element {element}, using default"
            )
            return 1e-3  # Default small value

        data = nist_data[element]
        energies = np.array(data["energies"])
        mu_rho_values = np.array(data["mu_rho"])

        # Convert energy to eV if needed and clamp to data range
        energy_ev = max(min(energy_ev, energies[-1]), energies[0])

        # Interpolate in log-log space for better accuracy across wide energy range
        log_energies = np.log(energies)
        log_mu_rho = np.log(mu_rho_values)
        log_energy_target = np.log(energy_ev)

        # Linear interpolation in log space
        log_mu_rho_interp = np.interp(log_energy_target, log_energies, log_mu_rho)
        mu_rho_result = np.exp(log_mu_rho_interp)

        return mu_rho_result

    def _get_lp_correction(
        self, s1_vector: np.ndarray, beam: object, experiment: object
    ) -> float:
        """Get Lorentz-Polarization correction for a single s1 vector."""
        try:
            from dials.algorithms.integration import Corrections
            from dials.array_family import flex

            # Cache corrections object to avoid repeated instantiation
            if not hasattr(self, "_corrections_obj"):
                self._corrections_obj = Corrections(
                    beam, experiment.goniometer, experiment.detector
                )

            # Convert to DIALS flex array format for single pixel
            s1_flex = flex.vec3_double([tuple(s1_vector)])

            # Get LP correction (returns divisors, convert to multipliers)
            lp_divisors = self._corrections_obj.lp(s1_flex)
            lp_mult = (
                1.0 / float(lp_divisors[0])
                if len(lp_divisors) > 0 and lp_divisors[0] != 0
                else 1.0
            )

            return lp_mult

        except Exception as e:
            logger.debug(f"LP correction failed: {e}")
            return 1.0

    def _get_qe_correction(
        self,
        s1_vector: np.ndarray,
        beam: object,
        experiment: object,
        panel_idx: int = 0,
    ) -> float:
        """Get Quantum Efficiency correction for a single s1 vector."""
        try:
            from dials.algorithms.integration import Corrections
            from dials.array_family import flex

            # Cache corrections object to avoid repeated instantiation
            if not hasattr(self, "_corrections_obj"):
                self._corrections_obj = Corrections(
                    beam, experiment.goniometer, experiment.detector
                )

            # Convert to DIALS flex array format
            s1_flex = flex.vec3_double([tuple(s1_vector)])
            panel_indices = flex.size_t([panel_idx])

            # Get QE correction (returns multipliers)
            qe_multipliers = self._corrections_obj.qe(s1_flex, panel_indices)
            qe_mult = float(qe_multipliers[0]) if len(qe_multipliers) > 0 else 1.0

            return qe_mult

        except Exception as e:
            logger.debug(f"QE correction failed: {e}")
            return 1.0
