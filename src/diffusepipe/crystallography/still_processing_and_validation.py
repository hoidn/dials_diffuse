"""
Still processor and validation component for orchestrating per‑still DIALS
processing and geometric validation.

This module provides a higher‑level interface for processing individual still
images using a DIALS adapter and performing geometric model validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np  # Used only for simple statistics in the plot helper

from diffusepipe.adapters.dials_sequence_process_adapter import (
    DIALSSequenceProcessAdapter,
)
from diffusepipe.adapters.dials_stills_process_adapter import (
    DIALSStillsProcessAdapter,
)
from diffusepipe.utils.cbf_utils import get_angle_increment_from_cbf
from diffusepipe.crystallography.q_consistency_checker import QConsistencyChecker
from diffusepipe.exceptions import ConfigurationError, DIALSError, DataValidationError
from diffusepipe.types.types_IDL import (
    DIALSStillsProcessConfig,
    ExtractionConfig,
    OperationOutcome,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#                               DATA CLASSES
# -----------------------------------------------------------------------------
class ValidationMetrics:
    """Container for validation metrics and results."""

    def __init__(self) -> None:
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

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise metrics so they can be placed into an OperationOutcome."""

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


# -----------------------------------------------------------------------------
#                           MODEL VALIDATOR CLASS
# -----------------------------------------------------------------------------
class ModelValidator:
    """Performs geometry checks: PDB match and Q‑vector consistency."""

    def __init__(self) -> None:  # noqa: D401
        self.q_checker = QConsistencyChecker()

    # ------------------------------------------------------------------
    def validate_geometry(
        self,
        experiment: object,
        reflections: object,
        *,
        external_pdb_path: Optional[str] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, ValidationMetrics]:
        """Run all validation sub‑checks and collate the results. Primary internal check is Q-vector consistency."""

        metrics = ValidationMetrics()

        # ---------------- PDB check ----------------
        cell_len_tol = extraction_config.cell_length_tol if extraction_config else 0.02
        cell_ang_tol = extraction_config.cell_angle_tol if extraction_config else 2.0
        orient_tol_deg = (
            extraction_config.orient_tolerance_deg if extraction_config else 5.0
        )

        if external_pdb_path and Path(external_pdb_path).exists():
            (
                metrics.pdb_cell_passed,
                metrics.pdb_orientation_passed,
                metrics.misorientation_angle_vs_pdb,
            ) = self._check_pdb_consistency(
                experiment,
                external_pdb_path,
                cell_len_tol,
                cell_ang_tol,
                orient_tol_deg,
            )

        # --------------- Q‑vector check -------------
        q_tol = (
            extraction_config.q_consistency_tolerance_angstrom_inv
            if extraction_config
            else 0.01
        )
        metrics.q_consistency_passed, stats = self._check_q_consistency(
            experiment, reflections, q_tol
        )
        metrics.mean_delta_q_mag = stats.get("mean")
        metrics.max_delta_q_mag = stats.get("max")
        metrics.median_delta_q_mag = stats.get("median")
        metrics.num_reflections_tested = stats.get("count", 0)

        # --------------- Diagnostics ---------------
        if output_dir:
            try:
                metrics.plot_paths = self._generate_diagnostic_plots(
                    stats, external_pdb_path, output_dir
                )
                metrics.validation_plots_generated = True
            except Exception as exc:  # pragma: no cover – best‑effort helper
                logger.warning("Plot generation failed: %s", exc)

        # --------------- Overall pass/fail ----------
        validation_passed = (
            metrics.q_consistency_passed is True
            and metrics.num_reflections_tested > 0          # NEW – guard against empty check
            and (metrics.pdb_cell_passed is not False)
            and (metrics.pdb_orientation_passed is not False)
        )
        logger.info("Geometric validation result: %s", validation_passed)
        return validation_passed, metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _check_pdb_consistency(
        experiment: object,
        pdb_path: str,
        cell_length_tol: float,
        cell_angle_tol: float,
        orient_tolerance_deg: float,
    ) -> Tuple[bool, bool, Optional[float]]:
        """Stub PDB comparison – always passes until fully implemented."""

        logger.info("Checking PDB consistency against %s", pdb_path)
        
        try:
            # Step 1: Load PDB crystal symmetry
            from iotbx import pdb
            from scitbx import matrix
            
            pdb_input = pdb.input(file_name=pdb_path)
            pdb_crystal_symmetry = pdb_input.crystal_symmetry()
            
            if pdb_crystal_symmetry is None:
                logger.warning("PDB file %s has no crystal symmetry information, skipping PDB consistency check", pdb_path)
                return True, True, None  # Pass if PDB lacks symmetry
            
            pdb_uc = pdb_crystal_symmetry.unit_cell()
            pdb_sg = pdb_crystal_symmetry.space_group()
            
            # Step 2: Get experiment crystal symmetry
            exp_crystal = experiment.crystal
            exp_uc = exp_crystal.get_unit_cell()
            exp_sg = exp_crystal.get_space_group()
            
            # Step 3: Compare unit cells
            cell_passed = exp_uc.is_similar_to(
                pdb_uc, 
                relative_length_tolerance=cell_length_tol,
                absolute_angle_tolerance=cell_angle_tol
            )
            
            # Log comparison details
            logger.info("Unit cell comparison:")
            logger.info("  Experiment: %s", exp_uc.parameters())
            logger.info("  PDB: %s", pdb_uc.parameters())
            logger.info("  Cell similarity: %s (tol: length=%g, angle=%g°)", 
                       cell_passed, cell_length_tol, cell_angle_tol)
            
            # Step 4: Compare space groups (informational)
            if exp_sg.type().number() != pdb_sg.type().number():
                logger.warning("Space group mismatch: experiment=%s, PDB=%s", 
                             exp_sg.type().lookup_symbol(), pdb_sg.type().lookup_symbol())
            
            # Step 5: Compare orientations
            A_dials = matrix.sqr(exp_crystal.get_A())
            B_pdb = matrix.sqr(pdb_uc.fractionalization_matrix()).transpose().inverse()
            
            # For PDB, assume conventional orientation (U_pdb = Identity), so A_pdb_ref = B_pdb
            misorientation_deg = ModelValidator._calculate_misorientation_static(A_dials, B_pdb)
            orientation_passed = misorientation_deg <= orient_tolerance_deg
            
            logger.info("Orientation comparison:")
            logger.info("  Misorientation angle: %.2f° (tolerance: %.2f°)", 
                       misorientation_deg, orient_tolerance_deg)
            logger.info("  Orientation similarity: %s", orientation_passed)
            
            return cell_passed, orientation_passed, misorientation_deg
            
        except Exception as e:
            logger.error("PDB consistency check failed: %s", e)
            return False, False, None

    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_misorientation_static(A1_matrix_sqr: 'matrix.sqr', A2_matrix_sqr: 'matrix.sqr') -> float:
        """Calculate misorientation angle between two A-matrices (UB matrices)."""
        
        from scitbx import matrix
        import numpy as np
        
        def angle_between_orientations(a_mat: 'matrix.sqr', b_mat: 'matrix.sqr') -> float:
            try:
                a_inv = a_mat.inverse()
            except RuntimeError:  # Singular matrix
                return 180.0
            r_ab = b_mat * a_inv
            trace_r = r_ab.trace()
            cos_angle = (trace_r - 1.0) / 2.0
            cos_angle_clipped = np.clip(cos_angle, -1.0, 1.0)  # Handle precision errors
            angle_rad = np.arccos(cos_angle_clipped)
            return np.degrees(angle_rad)
        
        # Compare A1 with A2 and A1 with -A2 (handles potential hand inversion)
        mis_direct = angle_between_orientations(A1_matrix_sqr, A2_matrix_sqr)
        A2_inverted_hand = matrix.sqr([-x for x in A2_matrix_sqr.elems])
        mis_inverted = angle_between_orientations(A1_matrix_sqr, A2_inverted_hand)
        
        return min(mis_direct, mis_inverted)

    # ------------------------------------------------------------------
    def _check_q_consistency(
        self, experiment: object, reflections: object, tolerance: float
    ) -> Tuple[bool, Dict[str, float]]:
        """Perform Q-vector consistency check by comparing model-derived q-vectors with those recalculated from observed pixel positions. Delegates to QConsistencyChecker."""
        return self.q_checker.check_q_consistency(experiment, reflections, tolerance)

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_diagnostic_plots(
        q_stats: Dict[str, float],
        pdb_path: Optional[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """Create very simple summary plots – keeps dependencies light."""

        plot_paths: Dict[str, str] = {}
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt

            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)

            # --- Δq histogram ----------------------------------------
            if q_stats.get("count", 0) > 0 and q_stats.get("mean") is not None:
                # Simulate a tiny histogram with mean/median/max markers
                fig, ax = plt.subplots(figsize=(6, 4))
                # Fake distribution: we only have summary stats, so just plot a bar
                ax.bar(0, q_stats["mean"], width=0.4, label="mean |Δq|")
                ax.bar(0.6, q_stats["max"], width=0.4, label="max |Δq|")
                ax.set_ylabel("Å⁻¹")
                ax.set_xticks([])
                ax.set_title("Q‑vector discrepancy summary")
                ax.legend()
                out_path = output / "q_consistency_summary.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=120)
                plt.close(fig)
                plot_paths["q_consistency"] = str(out_path)

            # --- PDB placeholder -------------------------------------
            if pdb_path:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(
                    0.5,
                    0.5,
                    "PDB comparison\nnot yet implemented",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                out_path = output / "pdb_comparison_placeholder.png"
                plt.savefig(out_path, dpi=120)
                plt.close(fig)
                plot_paths["pdb_consistency"] = str(out_path)

        except ImportError:
            logger.warning("matplotlib not available – skipping diagnostic plots")
        return plot_paths


# -----------------------------------------------------------------------------
#          ORCHESTRATOR (PROCESSING + VALIDATION) – PUBLIC ENTRY POINT
# -----------------------------------------------------------------------------
class StillProcessorAndValidatorComponent:
    """Run DIALS processing and then the geometry checks in one call."""

    def __init__(self) -> None:  # noqa: D401
        self.stills_adapter = DIALSStillsProcessAdapter()
        self.sequence_adapter = DIALSSequenceProcessAdapter()
        self.validator = ModelValidator()

    # ------------------------------------------------------------------
    def _determine_processing_route(
        self, 
        image_path: str, 
        config: DIALSStillsProcessConfig
    ) -> Tuple[str, object]:
        """
        Determine which adapter to use based on CBF data type detection.
        
        Implements Module 1.S.0: CBF Data Type Detection and Processing Route Selection.
        
        Args:
            image_path: Path to CBF image file
            config: Configuration that may override auto-detection
            
        Returns:
            Tuple of (processing_route, selected_adapter)
            where processing_route is "stills" or "sequence"
        """
        # Step 1: Check for force override
        if config.force_processing_mode:
            if config.force_processing_mode.lower() == "stills":
                logger.info(f"Force override: using stills processing for {image_path}")
                return "stills", self.stills_adapter
            elif config.force_processing_mode.lower() == "sequence":
                logger.info(f"Force override: using sequence processing for {image_path}")
                return "sequence", self.sequence_adapter
            else:
                logger.warning(f"Invalid force_processing_mode: {config.force_processing_mode}, falling back to auto-detection")
        
        # Step 2: Auto-detect from CBF header
        try:
            angle_increment = get_angle_increment_from_cbf(image_path)
            
            if angle_increment is not None:
                if angle_increment == 0.0:
                    logger.info(f"Auto-detected stills data (Angle_increment=0.0°) for {image_path}")
                    return "stills", self.stills_adapter
                elif angle_increment > 0.0:
                    logger.info(f"Auto-detected sequence data (Angle_increment={angle_increment}°) for {image_path}")
                    return "sequence", self.sequence_adapter
                else:
                    logger.warning(f"Unexpected negative Angle_increment ({angle_increment}°), defaulting to sequence processing")
                    return "sequence", self.sequence_adapter
            else:
                logger.warning(f"Could not determine Angle_increment from {image_path}, defaulting to sequence processing (safer)")
                return "sequence", self.sequence_adapter
                
        except Exception as e:
            logger.warning(f"CBF header parsing failed for {image_path}: {e}, defaulting to sequence processing")
            return "sequence", self.sequence_adapter

    # ------------------------------------------------------------------
    def process_and_validate_still(
        self,
        *,
        image_path: str,
        config: DIALSStillsProcessConfig,
        extraction_config: ExtractionConfig,
        base_experiment_path: Optional[str] = None,
        external_pdb_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> OperationOutcome:
        """Helper that chains adapter + validator and packs an OperationOutcome."""

        logger.info("Processing still image: %s", image_path)

        # -------------------- Module 1.S.0: CBF Data Type Detection and Processing Route Selection ---------------------
        processing_route, selected_adapter = self._determine_processing_route(image_path, config)
        logger.info(f"Selected processing route: {processing_route}")

        # -------------------- DIALS processing ---------------------
        try:
            exp, refl, success, log = selected_adapter.process_still(
                image_path=image_path,
                config=config,
                base_expt_path=base_experiment_path,
            )
        except Exception as exc:
            logger.error("DIALS processing raised: %s", exc)
            return OperationOutcome(
                status="FAILURE_DIALS_PROCESSING",
                message=str(exc),
                error_code="DIALS_PROCESSING_EXCEPTION",
                output_artifacts={},
            )

        if not success or exp is None or refl is None:
            return OperationOutcome(
                status="FAILURE_DIALS_PROCESSING",
                message="DIALS processing failed – see log_messages",
                error_code="DIALS_PROCESSING_FAILED",
                output_artifacts={"log_messages": log},
            )

        # -------------------- Validation ---------------------------
        passed, metrics = self.validator.validate_geometry(
            experiment=exp,
            reflections=refl,
            external_pdb_path=external_pdb_path,
            extraction_config=extraction_config,
            output_dir=output_dir,
        )

        outcome_status = "SUCCESS" if passed else "FAILURE_GEOMETRY_VALIDATION"
        return OperationOutcome(
            status=outcome_status,
            message="Processed and validated" if passed else "Validation failed",
            error_code=None if passed else "GEOMETRY_VALIDATION_FAILED",
            output_artifacts={
                "experiment": exp,
                "reflections": refl,
                "validation_passed": passed,
                "validation_metrics": metrics.to_dict(),
                "processing_route_used": processing_route,  # Include routing information
                "log_messages": log,
            },
        )

    # ------------------------------------------------------------------
    # Backward‑compat shortcut -----------------------------------------
    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_experiment_path: Optional[str] = None,
    ) -> OperationOutcome:  # noqa: D401 – keep old name
        """Legacy API: just run DIALS without validation."""
        # Use routing logic even for legacy API
        processing_route, selected_adapter = self._determine_processing_route(image_path, config)
        logger.info(f"Legacy API: Selected processing route: {processing_route}")
        
        exp, refl, success, log = selected_adapter.process_still(
            image_path=image_path,
            config=config,
            base_expt_path=base_experiment_path,
        )
        if success and exp is not None and refl is not None:
            return OperationOutcome(
                status="SUCCESS",
                message="DIALS processing only (legacy path)",
                error_code=None,
                output_artifacts={
                    "experiment": exp,
                    "reflections": refl,
                    "processing_route_used": processing_route,  # Include routing information
                    "log_messages": log,
                },
            )
        return OperationOutcome(
            status="FAILURE",
            message="DIALS processing failed (legacy path)",
            error_code="DIALS_PROCESSING_FAILED",
            output_artifacts={
                "processing_route_used": processing_route,  # Include routing information even for failures
                "log_messages": log,
            },
        )


# -----------------------------------------------------------------------------
#                           CONFIGURATION HELPERS
# -----------------------------------------------------------------------------
def create_default_config(
    phil_path: Optional[str] = None,
    enable_partiality: bool = True,
    enable_shoeboxes: bool = False,
    known_unit_cell: Optional[str] = None,
    known_space_group: Optional[str] = None,
) -> DIALSStillsProcessConfig:
    """Create a default DIALS stills process configuration."""
    
    return DIALSStillsProcessConfig(
        stills_process_phil_path=phil_path,
        force_processing_mode=None,  # Default to auto-detection
        calculate_partiality=enable_partiality,
        output_shoeboxes=enable_shoeboxes,
        known_unit_cell=known_unit_cell,
        known_space_group=known_space_group,
        spotfinder_threshold_algorithm="dispersion",
        min_spot_area=3,
    )


def create_default_extraction_config() -> ExtractionConfig:
    """Create a default extraction configuration with reasonable validation tolerances."""
    
    return ExtractionConfig(
        gain=1.0,
        cell_length_tol=0.02,  # 2%
        cell_angle_tol=2.0,    # 2 degrees
        orient_tolerance_deg=5.0,  # 5 degrees  
        q_consistency_tolerance_angstrom_inv=0.01,  # 0.01 Å⁻¹
        pixel_step=1,
        lp_correction_enabled=False,
        plot_diagnostics=True,
        verbose=False,
    )


# -----------------------------------------------------------------------------
# Back-compat shim — legacy code and tests expect this symbol to exist
# -----------------------------------------------------------------------------
class StillProcessorComponent(StillProcessorAndValidatorComponent):
    """
    Legacy alias preserved for external code that still does:

        from diffusepipe.crystallography.still_processing_and_validation \
            import StillProcessorComponent
    """
    # No extra behaviour; everything lives in the parent class.
    pass
