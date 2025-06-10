"""
Q-vector consistency checker for validating DIALS processing results.

This module provides functionality to check consistency between q-vectors calculated from 
Miller indices/crystal model (q_bragg) and q-vectors calculated from observed pixel 
positions and detector geometry (q_observed).
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class QConsistencyChecker:
    """
    Checks consistency between q_bragg (from Miller indices) and q_observed (from pixels).
    
    This class implements the complex Q-vector consistency validation logic that was
    previously embedded in the StillProcessorAndValidatorComponent.
    """

    def check_q_consistency(
        self, experiment: object, reflections: object, tolerance: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check consistency between q_bragg and q_pixel_recalculated.

        Args:
            experiment: DIALS Experiment object containing crystal, detector, beam models.
            reflections: DIALS reflection_table with Miller indices and positions.
            tolerance: Q-vector magnitude tolerance in Å⁻¹.

        Returns:
            Tuple of (passed, statistics_dict).
        """
        try:
            logger.info(
                "Performing Q-vector consistency check (aligning with consistency_checker.py logic)"
            )

            # DEBUG: Log all available columns in reflections table
            try:
                available_cols = list(reflections.keys()) if hasattr(reflections, 'keys') else []
                logger.debug(f"DEBUG: Available columns in reflections table: {available_cols}")
                logger.debug(f"DEBUG: Reflections table type: {type(reflections)}")
                logger.debug(f"DEBUG: Reflections table size: {len(reflections) if hasattr(reflections, '__len__') else 'unknown'}")
            except Exception as e:
                logger.error(f"DEBUG: Failed to inspect reflections table: {e}")

            # Essential columns for this check
            required_cols = ["miller_index", "panel"]
            missing_cols = []
            present_cols = []
            
            for col in required_cols:
                try:
                    if col in reflections:
                        present_cols.append(col)
                        logger.debug(f"DEBUG: Required column '{col}' is present")
                    else:
                        missing_cols.append(col)
                        logger.debug(f"DEBUG: Required column '{col}' is MISSING")
                except Exception as e:
                    logger.error(f"DEBUG: Error checking for column '{col}': {e}")
                    missing_cols.append(col)

            if missing_cols:
                logger.warning(
                    f"Missing required columns {missing_cols} for Q-vector consistency check."
                )
                logger.warning(f"Present required columns: {present_cols}")
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            # DEBUG: Log sample data from required columns
            try:
                miller_indices_sample = reflections["miller_index"][:min(3, len(reflections))] if len(reflections) > 0 else []
                panel_sample = reflections["panel"][:min(3, len(reflections))] if len(reflections) > 0 else []
                logger.debug(f"DEBUG: Sample miller_index values: {miller_indices_sample}")
                logger.debug(f"DEBUG: Sample panel values: {panel_sample}")
            except Exception as e:
                logger.error(f"DEBUG: Failed to sample required column data: {e}")

            # Find best available position column
            position_col_name = self._find_best_position_column(reflections)
            if position_col_name is None:
                logger.warning("No suitable reflection position column found.")
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            n_total = len(reflections)
            logger.debug(f"DEBUG: Total number of reflections: {n_total}")
            
            if n_total == 0:
                logger.warning("No reflections available for Q-vector consistency check.")
                return False, {"count": 0, "mean": None, "max": None, "median": None}

            # Process reflections and calculate delta_q values
            delta_q_magnitudes = self._process_reflections(
                experiment, reflections, position_col_name, n_total
            )

            # Calculate statistics and determine pass/fail
            return self._calculate_statistics(delta_q_magnitudes, tolerance)

        except Exception as e:
            logger.error(f"Q-vector consistency check failed: {e}")
            return False, {"count": 0, "mean": None, "max": None, "median": None}

    def _find_best_position_column(self, reflections: object) -> str:
        """Find the best available position column, prioritizing observed coordinates."""
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
        
        return position_col_name

    def _process_reflections(
        self, experiment: object, reflections: object, position_col_name: str, n_total: int
    ) -> list:
        """Process reflections and calculate delta_q magnitudes."""
        # Use a representative subset of reflections for testing
        n_test = min(n_total, 500)  # Test up to 500 reflections
        indices = list(range(n_total))
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        logger.debug(f"DEBUG: Will test {n_test} reflections out of {n_total} total")

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

        for i, idx in enumerate(test_indices):
            try:
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

                # Calculate q_bragg and q_observed
                q_bragg = self._calculate_q_bragg(experiment, hkl, reflections, idx, i)
                if q_bragg is None:
                    processing_stats["failed_q_calculation"] += 1
                    continue

                q_observed = self._calculate_q_observed(
                    reflections, position_col_name, idx, current_panel, current_beam, i
                )
                if q_observed is None:
                    processing_stats["failed_pixel_coords"] += 1
                    continue

                # Calculate delta_q magnitude
                delta_q = q_bragg - q_observed
                delta_q_mag = np.linalg.norm(delta_q)
                delta_q_magnitudes.append(delta_q_mag)
                processing_stats["processed_successfully"] += 1

                if i < 3:
                    logger.debug(f"DEBUG: Reflection {idx} - |Δq| = {delta_q_mag:.4f} Å⁻¹")

            except Exception as e:
                logger.debug(f"DEBUG: Unexpected error processing reflection {idx}: {e}")
                processing_stats["failed_other"] += 1
                continue

        # Log processing statistics
        logger.info(f"Q-consistency check processing stats: {processing_stats}")
        
        if processing_stats["processed_successfully"] == 0:
            logger.error("No valid reflections processed for Q-consistency check.")
            logger.error(f"Processing failed for all {len(test_indices)} attempted reflections")

        return delta_q_magnitudes

    def _calculate_q_bragg(self, experiment: object, hkl: tuple, reflections: object, idx: int, i: int) -> np.ndarray:
        """Calculate q_bragg from Miller indices and crystal model."""
        try:
            # Use DIALS' model-based q-vector calculation
            if 's1' in reflections:
                s1_vec = np.array(reflections['s1'][idx])
                s0_vec = np.array(experiment.beam.get_s0())
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
            
            return q_bragg
            
        except Exception as e:
            logger.debug(f"DEBUG: Failed to calculate q_bragg for reflection {idx}: {e}")
            return None

    def _calculate_q_observed(
        self, reflections: object, position_col_name: str, idx: int, 
        current_panel: object, current_beam: object, i: int
    ) -> np.ndarray:
        """Calculate q_observed from pixel positions and detector geometry."""
        try:
            pixel_coords_for_lab = self._extract_pixel_coordinates(
                reflections, position_col_name, idx, current_panel, i
            )
            
            if pixel_coords_for_lab is None:
                logger.debug(f"Reflection {idx}: No valid pixel coordinates available. Skipping.")
                return None

            # Calculate lab coordinates from pixel position
            lab_coord = current_panel.get_pixel_lab_coord(pixel_coords_for_lab)
            lab_coord_np = np.array(lab_coord.elems)
            
            if i < 3:
                logger.debug(f"DEBUG: Reflection {idx} - lab coordinate: {lab_coord_np}")

            # Calculate q_observed (s1 - s0 from pixel position)
            s0_vec = np.array(current_beam.get_s0())
            s1_from_pixel = lab_coord_np / np.linalg.norm(lab_coord_np) * np.linalg.norm(s0_vec)
            q_observed = s1_from_pixel - s0_vec
            
            if i < 3:
                logger.debug(f"DEBUG: Reflection {idx} - q_observed: {q_observed} (mag: {np.linalg.norm(q_observed):.4f})")

            return q_observed

        except Exception as e:
            logger.debug(f"DEBUG: Failed to calculate q_observed for reflection {idx}: {e}")
            return None

    def _extract_pixel_coordinates(
        self, reflections: object, position_col_name: str, idx: int, current_panel: object, i: int
    ) -> tuple:
        """Extract pixel coordinates from reflection data."""
        pixel_coords_for_lab = None

        if position_col_name in ["xyzcal.mm", "xyzobs.mm.value"]:
            try:
                mm_pos = reflections[position_col_name][idx]
                x_mm, y_mm = mm_pos[0], mm_pos[1]
                
                if i < 3:
                    logger.debug(f"DEBUG: Reflection {idx} - mm position: ({x_mm}, {y_mm})")
                
                pixel_coords_for_lab = current_panel.millimeter_to_pixel((x_mm, y_mm))
                
                if i < 3:
                    logger.debug(f"DEBUG: Reflection {idx} - converted to pixel coords: {pixel_coords_for_lab}")
            except RuntimeError as e:
                logger.debug(f"Reflection {idx}: mm_to_pixel conversion failed ({e}). Trying next fallback.")
            except Exception as e:
                logger.debug(f"DEBUG: Unexpected error in mm coordinate processing for reflection {idx}: {e}")

        if pixel_coords_for_lab is None and position_col_name in ["xyzcal.px", "xyzobs.px.value"]:
            try:
                px_pos_data = reflections[position_col_name][idx]
                pixel_coords_for_lab = (px_pos_data[0], px_pos_data[1])
                
                if i < 3:
                    logger.debug(f"DEBUG: Reflection {idx} - direct pixel coords: {pixel_coords_for_lab}")
            except Exception as e:
                logger.debug(f"DEBUG: Failed to extract pixel coordinates from {position_col_name} for reflection {idx}: {e}")

        return pixel_coords_for_lab

    def _calculate_statistics(self, delta_q_magnitudes: list, tolerance: float) -> Tuple[bool, Dict[str, float]]:
        """Calculate statistics from delta_q magnitudes and determine pass/fail."""
        n_processed = len(delta_q_magnitudes)
        
        if n_processed == 0:
            return False, {"count": 0, "mean": None, "max": None, "median": None}

        # Calculate statistics
        delta_q_array = np.array(delta_q_magnitudes)
        mean_delta_q = float(np.mean(delta_q_array))
        max_delta_q = float(np.max(delta_q_array))
        median_delta_q = float(np.median(delta_q_array))

        # Determine pass/fail based on tolerance
        passed = mean_delta_q <= tolerance

        logger.info(f"Q-consistency check results:")
        logger.info(f"  Reflections processed: {n_processed}")
        logger.info(f"  Mean |Δq|: {mean_delta_q:.4f} Å⁻¹")
        logger.info(f"  Max |Δq|: {max_delta_q:.4f} Å⁻¹")
        logger.info(f"  Median |Δq|: {median_delta_q:.4f} Å⁻¹")
        logger.info(f"  Tolerance: {tolerance:.4f} Å⁻¹")
        logger.info(f"  Result: {'PASS' if passed else 'FAIL'}")

        stats = {
            "count": n_processed,
            "mean": mean_delta_q,
            "max": max_delta_q,
            "median": median_delta_q
        }

        return passed, stats