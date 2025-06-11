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
        mask_total_2d: Optional[tuple] = None,
    ) -> OperationOutcome:
        """
        Extract diffuse scattering data from a still image.

        Args:
            inputs: Input file paths including CBF, experiment, and PDB
            config: Extraction configuration parameters
            output_npz_path: Path for output NPZ file
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
            validation_result = self._validate_inputs(inputs, config, output_npz_path, mask_total_2d)
            if validation_result.status != "SUCCESS":
                return validation_result

            # 2. Load data
            logger.info("Loading input data")
            experiment, image_data, total_mask, pdb_data = self._load_data(
                inputs, mask_total_2d
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
            q_vectors, intensities, sigmas = self._process_pixels(
                experiment, image_data, total_mask, config
            )

            if len(q_vectors) == 0:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="ProcessingError",
                    message="No pixels passed filtering criteria",
                )

            # 5. Save output
            logger.info(f"Saving {len(q_vectors)} data points to {output_npz_path}")
            self._save_output(q_vectors, intensities, sigmas, output_npz_path)

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
        self, inputs: ComponentInputFiles, mask_total_2d: Optional[tuple] = None
    ) -> Tuple[object, np.ndarray, np.ndarray, Optional[object]]:
        """Load all required data files."""
        try:
            # Load DIALS experiment
            from dxtbx.model import ExperimentList

            experiment_list = ExperimentList.from_file(inputs.dials_expt_path)
            if len(experiment_list) == 0:
                raise ValueError("No experiments found in DIALS experiment file")
            experiment = experiment_list[0]

            # Validate experiment has required models
            if experiment.beam is None:
                raise ValueError("DIALS experiment missing beam model")
            if experiment.detector is None:
                raise ValueError("DIALS experiment missing detector model")
            if experiment.crystal is None:
                raise ValueError("DIALS experiment missing crystal model")

            # Load image data
            from dxtbx.imageset import ImageSetFactory

            imageset = ImageSetFactory.from_template(
                inputs.cbf_image_path, image_range=(1, 1)
            )
            image_data = imageset.get_raw_data(0)  # Get first (and only) image

            # Convert to numpy array if needed
            if hasattr(image_data, "__len__") and len(image_data) > 1:
                # Multi-panel detector - concatenate or handle appropriately
                # For now, assume single panel
                image_data = image_data[0].as_numpy_array()
            else:
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process detector pixels to extract q-vectors, intensities, and errors."""
        q_vectors_list = []
        intensities_list = []
        sigmas_list = []

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
                    lab_coord = panel.get_pixel_lab_coord((fast_idx, slow_idx))
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
        else:
            q_vectors = np.empty((0, 3))
            intensities = np.empty(0)
            sigmas = np.empty(0)

        return q_vectors, intensities, sigmas

    def _save_output(
        self,
        q_vectors: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        output_path: str,
    ):
        """Save extracted data to NPZ file."""
        try:
            np.savez_compressed(
                output_path, q_vectors=q_vectors, intensities=intensities, sigmas=sigmas
            )
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

        Implements Beer-Lambert law: Attenuation = exp(-μ_air * path_length)
        Returns: Air_mult = 1.0 / Attenuation
        """
        try:
            # Get X-ray energy from wavelength
            wavelength = beam.get_wavelength()  # Angstroms
            energy_ev = 12398.4 / wavelength  # eV

            # Path length from sample to pixel (assuming sample at origin)
            path_length = np.linalg.norm(lab_coord) / 1000.0  # Convert mm to meters

            # Calculate linear attenuation coefficient for air at this energy
            # Using approximate values for dry air at STP
            # This is a simplified implementation - production code should use
            # tabulated values from NIST or libraries like xraylib
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

        This is a simplified implementation using approximate values.
        Production code should use tabulated values from NIST or libraries like xraylib.

        Args:
            energy_ev: X-ray energy in eV

        Returns:
            Linear attenuation coefficient in m⁻¹
        """
        try:
            # Simplified approximation for air based on temperature and pressure
            # Based on mass attenuation coefficients for N₂ (78%), O₂ (21%), Ar (1%)

            # Calculate air density based on ideal gas law
            # ρ = (P × M) / (R × T), where M is molar mass of air (~29 g/mol)
            # At STP: ρ = 1.225 kg/m³ = 0.001225 g/cm³
            air_density_stp = 0.001225  # g/cm³ at STP (273.15 K, 1 atm)
            air_density = (
                air_density_stp * (pressure_atm / 1.0) * (273.15 / temperature_k)
            )

            # Very rough approximation: μ/ρ ≈ C × λ³ for λ in Angstroms
            # This is NOT accurate for production use!
            wavelength_angstrom = 12398.4 / energy_ev

            if energy_ev > 1000:  # Above 1 keV, use simple power law
                # Mass attenuation coefficient (very rough approximation)
                mu_over_rho = 0.1 * (wavelength_angstrom**2.8)  # cm²/g

                # Convert to linear attenuation coefficient
                mu_linear = mu_over_rho * air_density  # cm⁻¹
                mu_linear_m = mu_linear * 100  # m⁻¹
            else:
                # For very low energies, assume minimal attenuation
                mu_linear_m = 0.01 * (
                    air_density / air_density_stp
                )  # Scale with density

            return mu_linear_m

        except Exception as e:
            logger.debug(f"Air attenuation coefficient calculation failed: {e}")
            return 0.001  # Very small default value

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
