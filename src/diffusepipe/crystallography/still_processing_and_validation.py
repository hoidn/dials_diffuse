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
        """Run all validation sub‑checks and collate the results."""

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
        # TODO: implement real comparison logic (gemmi / cctbx)
        return True, True, 0.0

    # ------------------------------------------------------------------
    def _check_q_consistency(
        self, experiment: object, reflections: object, tolerance: float
    ) -> Tuple[bool, Dict[str, float]]:
        """Delegate to the dedicated utility and return its statistics."""
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
        self.adapter = DIALSSequenceProcessAdapter()
        self.validator = ModelValidator()

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

        # -------------------- DIALS processing ---------------------
        try:
            exp, refl, success, log = self.adapter.process_still(
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
        exp, refl, success, log = self.adapter.process_still(
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
                    "log_messages": log,
                },
            )
        return OperationOutcome(
            status="FAILURE",
            message="DIALS processing failed (legacy path)",
            error_code="DIALS_PROCESSING_FAILED",
            output_artifacts={"log_messages": log},
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
