"""
Still processor and validation component for orchestrating per-still DIALS processing and geometric validation.

This module provides a higher-level interface for processing individual still images
using the DIALSStillsProcessAdapter and performing geometric model validation,
implementing Module 1.S.1 from the plan.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from diffusepipe.adapters.dials_sequence_process_adapter import (
    DIALSSequenceProcessAdapter,
)
from diffusepipe.types.types_IDL import (
    DIALSStillsProcessConfig,
    OperationOutcome,
    ExtractionConfig,
)
from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError
from diffusepipe.crystallography.q_consistency_checker import QConsistencyChecker

logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Container for validation metrics and results."""

    def __init__(self):
        self.pdb_cell_passed: Optional[bool] = None
        self.pdb_orientation_passed: Optional[bool] = None
        self.q_consistency_passed: Optional[bool] = None
        self.mean_delta_q_mag: Optional[float] = None
        self.max_delta_q_mag: Optional[float] = None
        self.median_delta_q_mag: Optional[float] = None
        self.misorientation_angle_vs_pdb: Optional[float] = None
        self.num_reflections_tested: int = 0
        self.validation_plots_generated: bool = False
        self.plot_paths: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "pdb_cell_passed": self.pdb_cell_passed,
            "pdb_orientation_passed": self.pdb_orientation_passed,
            "q_consistency_passed": self.q_consistency_passed,
            "mean_delta_q_mag": self.mean_delta_q_mag,
            "max_delta_q_mag": self.max_delta_q_mag,
            "median_delta_q_mag": self.median_delta_q_mag,
            "misorientation_angle_vs_pdb": self.misorientation_angle_vs_pdb,
            "num_reflections_tested": self.num_reflections_tested,
            "validation_plots_generated": self.validation_plots_generated,
            "plot_paths": self.plot_paths,
        }


class ModelValidator:
    """
    Component for performing geometric model consistency checks.

    Implements Sub-Module 1.S.1.Validation from the plan, including PDB consistency
    checks and internal Q-vector consistency validation.
    """

    def __init__(self):
        """Initialize the model validator."""
        self.q_checker = QConsistencyChecker()

    def validate_geometry(
        self,
        experiment: object,
        reflections: object,
        external_pdb_path: Optional[str] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, ValidationMetrics]:
        """
        Perform geometric consistency checks on DIALS processing results.

        Uses pixel-based position consistency checking instead of complex q-vector calculations.

        Args:
            experiment: DIALS Experiment object containing crystal model
            reflections: DIALS reflection_table with indexed spots and pixel coordinates
            external_pdb_path: Optional path to reference PDB file
            extraction_config: Configuration containing tolerance parameters (including pixel_position_tolerance_px)
            output_dir: Optional directory for saving diagnostic plots

        Returns:
            Tuple of (validation_passed, ValidationMetrics)
        """
        metrics = ValidationMetrics()

        try:
            # Set default tolerances if config not provided
            if extraction_config:
                cell_length_tol = extraction_config.cell_length_tol
                cell_angle_tol = extraction_config.cell_angle_tol
                orient_tolerance_deg = extraction_config.orient_tolerance_deg
            else:
                # Default tolerances
                cell_length_tol = 0.02  # 2%
                cell_angle_tol = 2.0  # 2 degrees
                orient_tolerance_deg = 5.0  # 5 degrees

            # 1. PDB Consistency Checks (if external_pdb_path provided)
            if external_pdb_path and Path(external_pdb_path).exists():
                try:
                    pdb_cell_passed, pdb_orient_passed, misorientation_angle = (
                        self._check_pdb_consistency(
                            experiment,
                            external_pdb_path,
                            cell_length_tol,
                            cell_angle_tol,
                            orient_tolerance_deg,
                        )
                    )
                    metrics.pdb_cell_passed = pdb_cell_passed
                    metrics.pdb_orientation_passed = pdb_orient_passed
                    metrics.misorientation_angle_vs_pdb = misorientation_angle
                except Exception as e:
                    logger.warning(f"PDB consistency check failed: {e}")
                    metrics.pdb_cell_passed = False
                    metrics.pdb_orientation_passed = False

            # 2. Q-Vector Consistency Check
            try:
                q_tolerance = (
                    extraction_config.q_consistency_tolerance_angstrom_inv
                    if extraction_config
                    else 0.01
                )
                q_passed, delta_stats = self._check_q_consistency(
                    experiment, reflections, q_tolerance
                )
                metrics.q_consistency_passed = q_passed
                metrics.mean_delta_q_mag = delta_stats.get("mean")
                metrics.max_delta_q_mag = delta_stats.get("max")
                metrics.median_delta_q_mag = delta_stats.get("median")
                metrics.num_reflections_tested = delta_stats.get("count", 0)
            except Exception as e:
                logger.error(f"Q-vector consistency check failed: {e}")
                metrics.q_consistency_passed = False

            # 3. Generate diagnostic plots if output directory provided
            if output_dir:
                try:
                    plot_paths = self._generate_diagnostic_plots(
                        experiment, reflections, external_pdb_path, output_dir
                    )
                    metrics.validation_plots_generated = True
                    metrics.plot_paths = plot_paths
                except Exception as e:
                    logger.warning(f"Failed to generate diagnostic plots: {e}")

            # Determine overall validation result
            validation_passed = True

            # Check PDB results if they were performed
            if metrics.pdb_cell_passed is not None:
                validation_passed = validation_passed and metrics.pdb_cell_passed
            if metrics.pdb_orientation_passed is not None:
                validation_passed = validation_passed and metrics.pdb_orientation_passed

            # Check Q-consistency (always performed)
            if metrics.q_consistency_passed is not None:
                validation_passed = validation_passed and metrics.q_consistency_passed
            else:
                validation_passed = False

            logger.info(f"Geometric validation result: {validation_passed}")
            return validation_passed, metrics

        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return False, metrics

    def _check_pdb_consistency(
        self,
        experiment: object,
        pdb_path: str,
        cell_length_tol: float,
        cell_angle_tol: float,
        orient_tolerance_deg: float,
    ) -> Tuple[bool, bool, Optional[float]]:
        """
        Check consistency between DIALS crystal model and reference PDB.

        Returns:
            Tuple of (cell_passed, orientation_passed, misorientation_angle)
        """
        try:
            # This is a placeholder implementation - in real use, would need:
            # 1. PDB parser (could use gemmi, BioPython, or cctbx)
            # 2. Unit cell comparison logic
            # 3. Orientation matrix comparison

            logger.info(f"Checking PDB consistency against {pdb_path}")

            # For now, implement basic structure that would be filled in
            # with actual PDB reading and comparison logic
            crystal = experiment.crystal
            unit_cell = crystal.get_unit_cell()

            # Extract unit cell parameters
            a, b, c, alpha, beta, gamma = unit_cell.parameters()

            # Placeholder: In real implementation, would load PDB and compare
            # For now, assume validation passes (would need actual PDB reading)
            logger.warning(
                "PDB consistency check not fully implemented - assuming pass"
            )

            cell_passed = True  # Would implement actual comparison
            orientation_passed = True  # Would implement actual comparison
            misorientation_angle = 0.0  # Would calculate actual misorientation

            return cell_passed, orientation_passed, misorientation_angle

        except Exception as e:
            logger.error(f"PDB consistency check error: {e}")
            return False, False, None

    def _check_q_consistency(
        self, experiment: object, reflections: object, tolerance: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check consistency between q_bragg and q_observed using dedicated checker.
        
        Args:
            experiment: DIALS Experiment object containing crystal, detector, beam models.
            reflections: DIALS reflection_table with Miller indices and positions.
            tolerance: Q-vector magnitude tolerance in Å⁻¹.

        Returns:
            Tuple of (passed, statistics_dict).
        """
        return self.q_checker.check_q_consistency(experiment, reflections, tolerance)

    def _generate_diagnostic_plots(
        self,
        experiment: object,
        reflections: object,
        pdb_path: Optional[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Generate diagnostic plots for validation results.

        Returns:
            Dictionary of plot_name -> file_path
        """
        plot_paths = {}

        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # DEBUG: Log sample data from required columns
            try:
                miller_indices_sample = reflections["miller_index"][:min(3, len(reflections))] if len(reflections) > 0 else []
                panel_sample = reflections["panel"][:min(3, len(reflections))] if len(reflections) > 0 else []
                logger.debug(f"DEBUG: Sample miller_index values: {miller_indices_sample}")
                logger.debug(f"DEBUG: Sample panel values: {panel_sample}")
            except Exception as e:
                logger.error(f"DEBUG: Failed to sample required column data: {e}")

            # Determine best available position column, prioritizing OBSERVED coordinates
            # to avoid tautological comparison with model predictions
            pos_col_preference = [
                "xyzobs.px.value",
                "xyzobs.mm.value", 
                "xyzcal.px",
                "xyzcal.mm",
            ]
            position_col_name = None
            available_pos_cols = []
            
            for col_name in pos_col_preference:
                try:
                    if col_name in reflections:
                        available_pos_cols.append(col_name)
                        if position_col_name is None:
                            position_col_name = col_name
                            logger.info(f"Using position column: {position_col_name}")
                except Exception as e:
                    logger.error(f"DEBUG: Error checking position column '{col_name}': {e}")
            
            logger.debug(f"DEBUG: Available position columns: {available_pos_cols}")
            logger.debug(f"DEBUG: Selected position column: {position_col_name}")

            if position_col_name is None:
                logger.warning("No suitable reflection position column found.")
                logger.warning(f"Searched for: {pos_col_preference}")
                logger.warning(f"Available position columns: {available_pos_cols}")
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            n_total = len(reflections)
            logger.debug(f"DEBUG: Total number of reflections: {n_total}")
            
            if n_total == 0:
                logger.warning(
                    "No reflections available for Q-vector consistency check."
                )
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            # Use a representative subset of reflections for testing
            n_test = min(n_total, 500)  # Test up to 500 reflections
            indices = list(range(n_total))
            np.random.shuffle(indices)  # Ensure numpy is imported
            test_indices = indices[:n_test]
            logger.debug(f"DEBUG: Will test {n_test} reflections out of {n_total} total")
            logger.debug(f"DEBUG: First few test indices: {test_indices[:min(5, len(test_indices))]}")

            delta_q_magnitudes = []
            processing_stats = {
                "processed_successfully": 0,
                "failed_hkl_panel": 0,
                "failed_invalid_panel": 0,
                "failed_pixel_coords": 0,
                "failed_q_calculation": 0,
                "failed_other": 0
            }

            current_crystal = experiment.crystal
            current_beam = experiment.beam
            current_detector = experiment.detector
            
            logger.debug(f"DEBUG: Crystal type: {type(current_crystal)}")
            logger.debug(f"DEBUG: Beam type: {type(current_beam)}")
            logger.debug(f"DEBUG: Detector type: {type(current_detector)}")
            logger.debug(f"DEBUG: Number of detector panels: {len(current_detector) if hasattr(current_detector, '__len__') else 'unknown'}")

            for i, idx in enumerate(test_indices):
                try:
                    # DEBUG: Log progress for first few reflections
                    if i < 10:
                        logger.debug(f"DEBUG: Processing reflection {i+1}/{n_test} (original index {idx})")
                    
                    # Extract basic reflection data
                    try:
                        hkl = reflections["miller_index"][idx]
                        panel_id = int(round(reflections["panel"][idx]))
                        
                        if i < 3:
                            logger.debug(f"DEBUG: Reflection {idx} - HKL: {hkl}, Panel: {panel_id}")
                    except Exception as e:
                        logger.debug(f"DEBUG: Failed to extract HKL/panel for reflection {idx}: {e}")
                        processing_stats["failed_hkl_panel"] += 1
                        continue

                    if not (0 <= panel_id < len(current_detector)):
                        logger.debug(
                            f"Reflection {idx}: Invalid panel ID {panel_id} (detector has {len(current_detector)} panels). Skipping."
                        )
                        processing_stats["failed_invalid_panel"] += 1
                        continue

                    current_panel = current_detector[panel_id]

                    # 1. Calculate q_model from DIALS' prediction (s1 - s0)
                    try:
                        # Use DIALS' model-based q-vector calculation
                        # This represents the prediction based on crystal model + geometry
                        if 's1' in reflections:
                            s1_vec = np.array(reflections['s1'][idx])
                            s0_vec = np.array(current_beam.get_s0())
                            q_bragg = s1_vec - s0_vec
                        else:
                            # Fallback to manual calculation if s1 not available
                            from scitbx import matrix
                            hkl_vec = matrix.col(hkl)
                            A = matrix.sqr(experiment.crystal.get_A())
                            S = matrix.sqr(experiment.goniometer.get_setting_rotation())
                            F = matrix.sqr(experiment.goniometer.get_fixed_rotation())
                            C = matrix.sqr((1,0,0, 0,0,-1, 0,1,0))
                            R_lab = C * S * F
                            q_bragg_lab_scitbx = R_lab * A * hkl_vec
                            q_bragg = np.array(q_bragg_lab_scitbx.elems)
                        
                        if i < 3:
                            logger.debug(f"DEBUG: Reflection {idx} - q_bragg calculated: {q_bragg} (mag: {np.linalg.norm(q_bragg):.4f})")
                    except Exception as e:
                        logger.debug(f"DEBUG: Failed to calculate q_bragg for reflection {idx}: {e}")
                        processing_stats["failed_q_calculation"] += 1
                        continue

                    # 2. Calculate q_observed from OBSERVED pixel positions (data-driven)
                    pixel_coords_for_lab = None

                    if position_col_name in ["xyzcal.mm", "xyzobs.mm.value"]:
                        try:
                            mm_pos = reflections[position_col_name][idx]
                            x_mm, y_mm = mm_pos[0], mm_pos[1]
                            
                            if i < 3:
                                logger.debug(f"DEBUG: Reflection {idx} - mm position: ({x_mm}, {y_mm})")
                            
                            # millimeter_to_pixel returns (fast_pixel_idx, slow_pixel_idx)
                            pixel_coords_for_lab = current_panel.millimeter_to_pixel(
                                (x_mm, y_mm)
                            )
                            
                            if i < 3:
                                logger.debug(f"DEBUG: Reflection {idx} - converted to pixel coords: {pixel_coords_for_lab}")
                        except (
                            RuntimeError
                        ) as e:  # E.g. mm coordinates outside panel bounds
                            logger.debug(
                                f"Reflection {idx}: mm_to_pixel conversion failed ({e}). Trying next fallback."
                            )
                            pass  # Fall through to try other pixel sources if this fails
                        except Exception as e:
                            logger.debug(f"DEBUG: Unexpected error in mm coordinate processing for reflection {idx}: {e}")
                            pass

                    if pixel_coords_for_lab is None and position_col_name in [
                        "xyzcal.px",
                        "xyzobs.px.value",
                    ]:
                        try:
                            px_pos_data = reflections[position_col_name][idx]
                            pixel_coords_for_lab = (px_pos_data[0], px_pos_data[1])
                            
                            if i < 3:
                                logger.debug(f"DEBUG: Reflection {idx} - direct pixel coords: {pixel_coords_for_lab}")
                        except Exception as e:
                            logger.debug(f"DEBUG: Failed to extract pixel coordinates from {position_col_name} for reflection {idx}: {e}")

                    if pixel_coords_for_lab is None:
                        # If specific position_col_name failed, try a hard fallback if not already tried
                        if (
                            "xyzobs.px.value" in reflections
                            and position_col_name != "xyzobs.px.value"
                        ):
                            try:
                                px_pos_data = reflections["xyzobs.px.value"][idx]
                                pixel_coords_for_lab = (px_pos_data[0], px_pos_data[1])
                                logger.debug(
                                    f"Reflection {idx}: Fell back to 'xyzobs.px.value': {pixel_coords_for_lab}"
                                )
                            except Exception as e:
                                logger.debug(f"DEBUG: Fallback to xyzobs.px.value failed for reflection {idx}: {e}")
                        elif (
                            "xyzcal.px" in reflections
                            and position_col_name != "xyzcal.px"
                        ):
                            try:
                                px_pos_data = reflections["xyzcal.px"][idx]
                                pixel_coords_for_lab = (px_pos_data[0], px_pos_data[1])
                                logger.debug(f"Reflection {idx}: Fell back to 'xyzcal.px': {pixel_coords_for_lab}")
                            except Exception as e:
                                logger.debug(f"DEBUG: Fallback to xyzcal.px failed for reflection {idx}: {e}")

                    if pixel_coords_for_lab is None:
                        logger.debug(
                            f"Reflection {idx}: Could not obtain valid pixel coordinates after all fallbacks. Skipping."
                        )
                        processing_stats["failed_pixel_coords"] += 1
                        continue

                    # Ensure pixel_coords_for_lab are (fast_px, slow_px)
                    fast_px, slow_px = pixel_coords_for_lab[0], pixel_coords_for_lab[1]

                    # panel.get_pixel_lab_coord expects (fast_pixel_coord, slow_pixel_coord)
                    try:
                        # Use the same approach as consistency_checker.py
                        pixel_lab_coord = np.array(current_panel.get_pixel_lab_coord((fast_px, slow_px)))

                        s0_vec = np.array(current_beam.get_s0())
                        k_magnitude = np.linalg.norm(s0_vec)  # This is 1/wavelength

                        scattered_direction_norm = np.linalg.norm(pixel_lab_coord)
                        if scattered_direction_norm < 1e-9:
                            logger.debug(
                                f"Reflection {idx}: Scattered direction norm is zero. Skipping."
                            )
                            processing_stats["failed_q_calculation"] += 1
                            continue

                        s1_unit_vec = pixel_lab_coord / scattered_direction_norm
                        s1_vec = s1_unit_vec * k_magnitude  # k_out_wavevector / (2*pi)

                        q_observed = s1_vec - s0_vec

                        # Calculate difference between model prediction and observed data
                        delta_q = q_bragg - q_observed
                        delta_q_mag = np.linalg.norm(delta_q)
                        delta_q_magnitudes.append(delta_q_mag)
                        processing_stats["processed_successfully"] += 1

                        if i < 3 and logger.isEnabledFor(
                            logging.DEBUG
                        ):  # Log first 3 processed from random sample
                            logger.debug(f"Reflection (original index {idx}):")
                            logger.debug(
                                f"  HKL: {hkl}, Panel: {panel_id}, Pixel Coords Used: ({fast_px:.2f}, {slow_px:.2f}) from {position_col_name}"
                            )
                            logger.debug(
                                f"  q_bragg          : {q_bragg.tolist()} (mag: {np.linalg.norm(q_bragg):.4f})"
                            )
                            logger.debug(
                                f"  q_observed       : {q_observed.tolist()} (mag: {np.linalg.norm(q_observed):.4f})"
                            )
                            logger.debug(f"  |Δq|             : {delta_q_mag:.6f} Å⁻¹")
                            
                    except Exception as e:
                        logger.debug(f"DEBUG: Failed to calculate q_observed for reflection {idx}: {e}")
                        processing_stats["failed_q_calculation"] += 1
                        continue

                except Exception as e:
                    logger.debug(
                        f"Failed to process reflection original index {idx} for Q-vector check: {e}"
                    )
                    processing_stats["failed_other"] += 1
                    continue

            # DEBUG: Log processing statistics
            logger.debug(f"DEBUG: Processing statistics:")
            logger.debug(f"  Successfully processed: {processing_stats['processed_successfully']}")
            logger.debug(f"  Failed HKL/panel extraction: {processing_stats['failed_hkl_panel']}")
            logger.debug(f"  Failed invalid panel ID: {processing_stats['failed_invalid_panel']}")
            logger.debug(f"  Failed pixel coordinate extraction: {processing_stats['failed_pixel_coords']}")
            logger.debug(f"  Failed Q-vector calculations: {processing_stats['failed_q_calculation']}")
            logger.debug(f"  Failed other reasons: {processing_stats['failed_other']}")
            logger.debug(f"  Total delta_q_magnitudes collected: {len(delta_q_magnitudes)}")

            if not delta_q_magnitudes:
                logger.error("No valid reflections processed for Q-consistency check.")
                logger.error(f"Processing failed for all {n_test} attempted reflections")
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            delta_q_array = np.array(delta_q_magnitudes)
            stats = {
                "mean": float(np.mean(delta_q_array)),
                "max": float(np.max(delta_q_array)),
                "median": float(np.median(delta_q_array)),
                "count": len(delta_q_magnitudes),
            }

            # Check against tolerance: mean |Δq| should be within tolerance,
            # and max |Δq| within a multiple (e.g., 5x) of the tolerance.
            passed = stats["mean"] <= tolerance and stats["max"] <= (tolerance * 5)

            logger.info(
                f"Q-vector consistency: mean_Δq = {stats['mean']:.6f} Å⁻¹, max_Δq = {stats['max']:.6f} Å⁻¹, "
                f"tolerance = {tolerance:.6f} Å⁻¹, n_tested={stats['count']}, passed = {passed}"
            )

            return passed, stats

        except Exception as e:
            logger.error(
                f"Overall Q-vector consistency check error: {e}", exc_info=True
            )
            return False, {"count": 0, "mean": None, "max": None, "median": None}

    def _generate_diagnostic_plots(
        self,
        experiment: object,
        reflections: object,
        pdb_path: Optional[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Generate diagnostic plots for validation results.

        Returns:
            Dictionary of plot_name -> file_path
        """
        plot_paths = {}

        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Placeholder plots - in real implementation would generate:
            # 1. Q-difference histogram
            # 2. Q-magnitude scatter plot
            # 3. Q-difference heatmap on detector

            # For now, create simple placeholder plots
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "Q-vector consistency diagnostic plot\n(Placeholder)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Q-Vector Consistency Check")

            plot_path = output_path / "q_consistency_diagnostic.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plot_paths["q_consistency"] = str(plot_path)

            if pdb_path:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    "PDB consistency diagnostic plot\n(Placeholder)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("PDB Consistency Check")

                plot_path = output_path / "pdb_consistency_diagnostic.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                plot_paths["pdb_consistency"] = str(plot_path)

            logger.info(f"Generated {len(plot_paths)} diagnostic plots in {output_dir}")

        except ImportError:
            logger.warning("matplotlib not available for diagnostic plots")
        except Exception as e:
            logger.error(f"Failed to generate diagnostic plots: {e}")

        return plot_paths


class StillProcessorAndValidatorComponent:
    """
    Component for processing individual still images through DIALS stills_process
    and performing geometric model validation.

    This component orchestrates the use of DIALSStillsProcessAdapter and ModelValidator
    to perform spot finding, indexing, refinement, integration, and validation for
    still diffraction images.
    """

    def __init__(self):
        """Initialize the still processor and validator component."""
        self.adapter = DIALSSequenceProcessAdapter()
        self.validator = ModelValidator()

    def process_and_validate_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        extraction_config: ExtractionConfig,
        base_experiment_path: Optional[str] = None,
        external_pdb_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> OperationOutcome:
        """
        Process a single still image using DIALS stills_process and validate the results.

        Args:
            image_path: Path to the CBF image file to process
            config: Configuration parameters for DIALS stills_process
            extraction_config: Configuration containing validation tolerances
            base_experiment_path: Optional path to base experiment file for geometry
            external_pdb_path: Optional path to reference PDB file for validation
            output_dir: Optional directory for saving diagnostic plots

        Returns:
            OperationOutcome with processing and validation results

        Behavior:
            - Uses DIALSStillsProcessAdapter to perform DIALS processing
            - If DIALS processing succeeds, performs geometric validation
            - Returns experiment, reflection objects, and validation results
            - If validation fails, marks status as FAILURE_GEOMETRY_VALIDATION
        """
        try:
            logger.info(f"Processing and validating still image: {image_path}")

            # Step 1: Process using DIALS adapter
            experiment, reflections, success, log_messages = self.adapter.process_still(
                image_path=image_path,
                config=config,
                base_expt_path=base_experiment_path,
            )

            if not success or experiment is None or reflections is None:
                return OperationOutcome(
                    status="FAILURE_DIALS_PROCESSING",
                    message=f"DIALS processing failed for {image_path}: {log_messages}",
                    error_code="DIALS_PROCESSING_FAILED",
                    output_artifacts={"log_messages": log_messages},
                )

            # Step 2: Perform geometric validation
            validation_passed, validation_metrics = self.validator.validate_geometry(
                experiment=experiment,
                reflections=reflections,
                external_pdb_path=external_pdb_path,
                extraction_config=extraction_config,
                output_dir=output_dir,
            )

            # Step 3: Create output artifacts
            output_artifacts = {
                "experiment": experiment,
                "reflections": reflections,
                "validation_passed": validation_passed,
                "validation_metrics": validation_metrics.to_dict(),
                "log_messages": log_messages,
            }

            if validation_passed:
                return OperationOutcome(
                    status="SUCCESS",
                    message=f"Successfully processed and validated still image {image_path}",
                    error_code=None,
                    output_artifacts=output_artifacts,
                )
            else:
                return OperationOutcome(
                    status="FAILURE_GEOMETRY_VALIDATION",
                    message=f"Geometric validation failed for {image_path}",
                    error_code="GEOMETRY_VALIDATION_FAILED",
                    output_artifacts=output_artifacts,
                )

        except (DIALSError, ConfigurationError, DataValidationError) as e:
            logger.error(f"Known error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=str(e),
                error_code=type(e).__name__.upper(),
                output_artifacts=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR",
                output_artifacts=None,
            )

    def validate_processing_outcome(self, outcome: OperationOutcome) -> bool:
        """
        Validate that a processing outcome contains expected artifacts including validation results.

        Args:
            outcome: OperationOutcome from process_and_validate_still

        Returns:
            True if outcome contains valid experiment, reflection, and validation artifacts
        """
        if outcome.status not in ["SUCCESS", "FAILURE_GEOMETRY_VALIDATION"]:
            return False

        if not outcome.output_artifacts:
            return False

        # Check for required artifacts
        required_keys = [
            "experiment",
            "reflections",
            "validation_passed",
            "validation_metrics",
        ]
        for key in required_keys:
            if key not in outcome.output_artifacts:
                logger.error(f"Missing required artifact: {key}")
                return False

        # For DIALS processing success, validation artifacts should be present
        if (
            outcome.output_artifacts["experiment"] is None
            or outcome.output_artifacts["reflections"] is None
        ):
            logger.error("Experiment or reflections artifact is None")
            return False

        # Validate that reflections contain partiality column
        reflections = outcome.output_artifacts["reflections"]
        try:
            if hasattr(reflections, "has_key") and not reflections.has_key(
                "partiality"
            ):
                logger.error("Reflection table missing 'partiality' column")
                return False
        except AttributeError:
            # Handle case where reflections is a mock object
            logger.warning(
                "Could not validate partiality column (possibly mock object)"
            )

        return True


# Backward compatibility - keep the original StillProcessorComponent
class StillProcessorComponent(StillProcessorAndValidatorComponent):
    """
    Backward-compatible wrapper that maintains the original StillProcessorComponent interface.
    """

    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_experiment_path: Optional[str] = None,
    ) -> OperationOutcome:
        """
        Process a single still image using DIALS stills_process (without validation).

        This method provides backward compatibility with the original interface.
        """
        try:
            logger.info(f"Processing still image: {image_path}")

            # Process using the adapter
            experiment, reflections, success, log_messages = self.adapter.process_still(
                image_path=image_path,
                config=config,
                base_expt_path=base_experiment_path,
            )

            if success and experiment is not None and reflections is not None:
                # Create output artifacts dictionary
                output_artifacts = {
                    "experiment": experiment,
                    "reflections": reflections,
                    "log_messages": log_messages,
                }

                return OperationOutcome(
                    status="SUCCESS",
                    message=f"Successfully processed still image {image_path}",
                    error_code=None,
                    output_artifacts=output_artifacts,
                )
            else:
                return OperationOutcome(
                    status="FAILURE",
                    message=f"DIALS processing failed for {image_path}: {log_messages}",
                    error_code="DIALS_PROCESSING_FAILED",
                    output_artifacts={"log_messages": log_messages},
                )

        except (DIALSError, ConfigurationError, DataValidationError) as e:
            logger.error(f"Known error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=str(e),
                error_code=type(e).__name__.upper(),
                output_artifacts=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR",
                output_artifacts=None,
            )


def create_default_config(
    phil_path: Optional[str] = None,
    enable_partiality: bool = True,
    enable_shoeboxes: bool = False,
    known_unit_cell: Optional[str] = None,
    known_space_group: Optional[str] = None,
) -> DIALSStillsProcessConfig:
    """
    Create a default DIALS stills process configuration.

    Args:
        phil_path: Optional path to PHIL configuration file
        enable_partiality: Whether to enable partiality calculation
        enable_shoeboxes: Whether to enable shoebox output
        known_unit_cell: Optional unit cell parameters as comma-separated string (a,b,c,alpha,beta,gamma)
        known_space_group: Optional space group symbol

    Returns:
        DIALSStillsProcessConfig with default settings
    """
    return DIALSStillsProcessConfig(
        stills_process_phil_path=phil_path,
        known_unit_cell=known_unit_cell,
        known_space_group=known_space_group,
        spotfinder_threshold_algorithm=None,
        min_spot_area=None,
        output_shoeboxes=enable_shoeboxes,
        calculate_partiality=enable_partiality,
    )


def create_default_extraction_config() -> ExtractionConfig:
    """
    Create a default extraction configuration with reasonable validation tolerances.

    Returns:
        ExtractionConfig with default validation settings
    """
    return ExtractionConfig(
        min_res=None,
        max_res=None,
        min_intensity=None,
        max_intensity=None,
        gain=1.0,
        cell_length_tol=0.02,  # 2%
        cell_angle_tol=2.0,  # 2 degrees
        orient_tolerance_deg=5.0,  # 5 degrees
        q_consistency_tolerance_angstrom_inv=0.01,  # 0.01 Å⁻¹
        pixel_position_tolerance_px=2.0,  # 2.0 pixels
        pixel_step=1,
        lp_correction_enabled=True,
        subtract_measured_background_path=None,
        subtract_constant_background_value=None,
        plot_diagnostics=False,
        verbose=False,
    )
