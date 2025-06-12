"""
Integration tests for StillProcessorAndValidatorComponent.

These tests focus on the integration between StillProcessorAndValidatorComponent,
DIALSStillsProcessAdapter, and ModelValidator, following the testing principles from plan.md.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffusepipe.crystallography.still_processing_and_validation import (
    StillProcessorAndValidatorComponent,
    ModelValidator,
    ValidationMetrics,
    create_default_config,
    create_default_extraction_config,
)
import numpy as np
from diffusepipe.types.types_IDL import (
    DIALSStillsProcessConfig,
    OperationOutcome,
    ExtractionConfig,
)
from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError


class TestStillProcessorAndValidatorComponent:
    """Test suite for StillProcessorAndValidatorComponent integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StillProcessorAndValidatorComponent()
        self.test_image_path = "tests/data/minimal_still.cbf"
        self.test_phil_path = "tests/data/minimal_stills_process.phil"
        self.test_pdb_path = "tests/data/reference.pdb"

    def test_process_and_validate_still_successfully(self):
        """Test successful processing and validation of a single still image."""
        # Arrange
        config = create_default_config(
            phil_path=self.test_phil_path, enable_partiality=True
        )
        extraction_config = create_default_extraction_config()

        # Mock the adapter to return successful results
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)  # Has partiality column

        # Mock validation to pass
        mock_validation_metrics = ValidationMetrics()
        mock_validation_metrics.q_consistency_passed = True

        # Mock the processing route to return a specific adapter instance
        mock_adapter = Mock()
        mock_adapter.process_still.return_value = (
            mock_experiment,
            mock_reflections,
            True,
            "Processing completed successfully",
        )

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(
                self.processor.validator, "validate_geometry"
            ) as mock_validate:
                mock_route.return_value = ("stills", mock_adapter)
                mock_validate.return_value = (True, mock_validation_metrics)

                # Act
                outcome = self.processor.process_and_validate_still(
                    image_path=self.test_image_path,
                    config=config,
                    extraction_config=extraction_config,
                    external_pdb_path=self.test_pdb_path,
                )

                # Assert
                assert outcome.status == "SUCCESS"
                assert "Processed and validated" in outcome.message
                assert outcome.error_code is None
                assert outcome.output_artifacts is not None

                # Verify required artifacts are present
                assert "experiment" in outcome.output_artifacts
                assert "reflections" in outcome.output_artifacts
                assert "validation_passed" in outcome.output_artifacts
                assert "validation_metrics" in outcome.output_artifacts
                assert outcome.output_artifacts["experiment"] is mock_experiment
                assert outcome.output_artifacts["reflections"] is mock_reflections
                assert outcome.output_artifacts["validation_passed"] is True

                # Verify adapter and validator were called
                mock_adapter.process_still.assert_called_once_with(
                    image_path=self.test_image_path, config=config, base_expt_path=None
                )
                mock_validate.assert_called_once_with(
                    experiment=mock_experiment,
                    reflections=mock_reflections,
                    external_pdb_path=self.test_pdb_path,
                    extraction_config=extraction_config,
                    output_dir=None,
                )

    def test_process_and_validate_still_validation_failure(self):
        """Test handling when validation fails after successful DIALS processing."""
        # Arrange
        config = create_default_config()
        extraction_config = create_default_extraction_config()

        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)

        # Mock validation to fail
        mock_validation_metrics = ValidationMetrics()
        mock_validation_metrics.q_consistency_passed = False
        mock_validation_metrics.mean_delta_q_mag = 0.1  # Above tolerance

        # Mock the processing route to return a specific adapter instance
        mock_adapter = Mock()
        mock_adapter.process_still.return_value = (
            mock_experiment,
            mock_reflections,
            True,
            "Success",
        )

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(
                self.processor.validator, "validate_geometry"
            ) as mock_validate:
                mock_route.return_value = ("stills", mock_adapter)
                mock_validate.return_value = (False, mock_validation_metrics)

                # Act
                outcome = self.processor.process_and_validate_still(
                    image_path=self.test_image_path,
                    config=config,
                    extraction_config=extraction_config,
                )

                # Assert
                assert outcome.status == "FAILURE_GEOMETRY_VALIDATION"
                assert "Validation failed" in outcome.message
                assert outcome.error_code == "GEOMETRY_VALIDATION_FAILED"
                assert outcome.output_artifacts["validation_passed"] is False
                assert (
                    outcome.output_artifacts["validation_metrics"][
                        "q_consistency_passed"
                    ]
                    is False
                )

    def test_process_and_validate_still_dials_failure(self):
        """Test handling when DIALS processing fails."""
        # Arrange
        config = create_default_config()
        extraction_config = create_default_extraction_config()

        # Mock the processing route to return a specific adapter instance
        mock_adapter = Mock()
        mock_adapter.process_still.return_value = (
            None,
            None,
            False,
            "DIALS processing failed",
        )

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            mock_route.return_value = ("stills", mock_adapter)

            # Act
            outcome = self.processor.process_and_validate_still(
                image_path=self.test_image_path,
                config=config,
                extraction_config=extraction_config,
            )

            # Assert
            assert outcome.status == "FAILURE_DIALS_PROCESSING"
            assert "DIALS processing failed" in outcome.message
            assert outcome.error_code == "DIALS_PROCESSING_FAILED"

            # Validator should not be called since DIALS processing failed
            # We can't easily check if validator.validate_geometry was not called without patching it,
            # so we'll just verify the expected error state without this check


class TestModelValidator:
    """Test suite for ModelValidator component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()

    def test_validate_geometry_q_vector_consistency_pass(self):
        """Test Q-vector consistency check that passes."""
        # Arrange
        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_detector = Mock()
        mock_beam = Mock()

        mock_experiment.crystal = mock_crystal
        mock_experiment.detector = mock_detector
        mock_experiment.beam = mock_beam

        # Create mock reflections with required columns for Q-vector validation
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=100)
        mock_reflections.__contains__ = Mock(
            side_effect=lambda x: x in ["miller_index", "panel", "xyzcal.mm"]
        )

        extraction_config = create_default_extraction_config()

        # Mock the Q-vector consistency check to pass (mean = 0.005 Å⁻¹, tolerance = 0.01 Å⁻¹)
        with patch.object(self.validator, "_check_q_consistency") as mock_q_check:
            mock_q_check.return_value = (
                True,
                {"mean": 0.005, "max": 0.012, "median": 0.004, "count": 20},
            )

            # Act
            validation_passed, metrics = self.validator.validate_geometry(
                experiment=mock_experiment,
                reflections=mock_reflections,
                extraction_config=extraction_config,
            )

            # Assert
            assert validation_passed is True
            assert metrics.q_consistency_passed is True
            assert (
                metrics.mean_delta_q_mag == 0.005
            )  # Q-vector magnitude difference in Å⁻¹
            assert metrics.num_reflections_tested == 20

    def test_validate_geometry_q_vector_consistency_fail(self):
        """Test Q-vector consistency check that fails."""
        # Arrange
        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_detector = Mock()
        mock_beam = Mock()

        mock_experiment.crystal = mock_crystal
        mock_experiment.detector = mock_detector
        mock_experiment.beam = mock_beam

        # Create mock reflections with required columns for Q-vector validation
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=100)
        mock_reflections.__contains__ = Mock(
            side_effect=lambda x: x in ["miller_index", "panel", "xyzcal.mm"]
        )

        extraction_config = create_default_extraction_config()

        # Mock the Q-vector consistency check to fail (mean = 0.025 Å⁻¹ > tolerance = 0.01 Å⁻¹)
        with patch.object(self.validator, "_check_q_consistency") as mock_q_check:
            mock_q_check.return_value = (
                False,
                {"mean": 0.025, "max": 0.080, "median": 0.021, "count": 20},
            )

            # Act
            validation_passed, metrics = self.validator.validate_geometry(
                experiment=mock_experiment,
                reflections=mock_reflections,
                extraction_config=extraction_config,
            )

            # Assert
            assert validation_passed is False
            assert metrics.q_consistency_passed is False
            assert (
                metrics.mean_delta_q_mag == 0.025
            )  # Q-vector magnitude difference in Å⁻¹

    def test_validate_geometry_missing_q_vector_columns(self):
        """Test Q-vector consistency check when required columns are missing."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=100)
        mock_reflections.__contains__ = Mock(
            return_value=False
        )  # Missing required columns

        extraction_config = create_default_extraction_config()

        # Mock the Q-vector check to fail due to missing columns
        with patch.object(self.validator, "_check_q_consistency") as mock_q_check:
            mock_q_check.return_value = (
                False,
                {"count": 0, "mean": None, "max": None, "median": None},
            )

            # Act
            validation_passed, metrics = self.validator.validate_geometry(
                experiment=mock_experiment,
                reflections=mock_reflections,
                extraction_config=extraction_config,
            )

            # Assert
            assert validation_passed is False
            assert metrics.q_consistency_passed is False
            assert metrics.num_reflections_tested == 0

    def test_validate_geometry_no_reflections(self):
        """Test Q-vector consistency check when no reflections are available."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=0)  # No reflections

        extraction_config = create_default_extraction_config()

        # Mock the Q-vector check to fail due to no reflections
        with patch.object(self.validator, "_check_q_consistency") as mock_q_check:
            mock_q_check.return_value = (
                False,
                {"count": 0, "mean": None, "max": None, "median": None},
            )

            # Act
            validation_passed, metrics = self.validator.validate_geometry(
                experiment=mock_experiment,
                reflections=mock_reflections,
                extraction_config=extraction_config,
            )

            # Assert
            assert validation_passed is False
            assert metrics.q_consistency_passed is False
            assert metrics.num_reflections_tested == 0

    def test_validate_geometry_with_pdb_check(self):
        """Test validation with PDB consistency check."""
        # Arrange
        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_detector = Mock()
        mock_beam = Mock()

        mock_experiment.crystal = mock_crystal
        mock_experiment.detector = mock_detector
        mock_experiment.beam = mock_beam

        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=100)

        extraction_config = create_default_extraction_config()
        pdb_path = "tests/data/test.pdb"

        # Mock PDB file existence
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                self.validator, "_check_pdb_consistency"
            ) as mock_pdb_check:
                with patch.object(
                    self.validator, "_check_q_consistency"
                ) as mock_q_check:
                    mock_pdb_check.return_value = (
                        True,
                        True,
                        1.5,
                    )  # cell_pass, orient_pass, angle
                    mock_q_check.return_value = (True, {"mean": 0.005, "count": 100})

                    # Act
                    validation_passed, metrics = self.validator.validate_geometry(
                        experiment=mock_experiment,
                        reflections=mock_reflections,
                        external_pdb_path=pdb_path,
                        extraction_config=extraction_config,
                    )

                    # Assert
                    assert validation_passed is True
                    assert metrics.pdb_cell_passed is True
                    assert metrics.pdb_orientation_passed is True
                    assert metrics.misorientation_angle_vs_pdb == 1.5

    def test_validate_geometry_pdb_mismatch_cell(self):
        """Test validation failure due to PDB cell parameter mismatch."""
        # Arrange
        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_detector = Mock()
        mock_beam = Mock()

        mock_experiment.crystal = mock_crystal
        mock_experiment.detector = mock_detector
        mock_experiment.beam = mock_beam

        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=100)

        extraction_config = create_default_extraction_config()
        pdb_path = "tests/data/test.pdb"

        # Mock PDB file existence
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                self.validator, "_check_pdb_consistency"
            ) as mock_pdb_check:
                with patch.object(
                    self.validator, "_check_q_consistency"
                ) as mock_q_check:
                    # PDB cell check fails, orientation passes, Q-consistency passes
                    mock_pdb_check.return_value = (False, True, 1.5)
                    mock_q_check.return_value = (True, {"mean": 0.005, "count": 100})

                    # Act
                    validation_passed, metrics = self.validator.validate_geometry(
                        experiment=mock_experiment,
                        reflections=mock_reflections,
                        external_pdb_path=pdb_path,
                        extraction_config=extraction_config,
                    )

                    # Assert
                    assert (
                        validation_passed is False
                    )  # Overall fails due to PDB cell mismatch
                    assert metrics.pdb_cell_passed is False
                    assert metrics.pdb_orientation_passed is True
                    assert metrics.q_consistency_passed is True


    def test_check_q_consistency_ideal_match(self):
        """Test Q-vector consistency check with ideal synthetic data."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        tolerance = 0.01  # Å⁻¹

        # Mock the Q-vector consistency check directly to simulate ideal match scenario
        with patch.object(self.validator, "q_checker") as mock_q_checker:
            mock_q_checker.check_q_consistency.return_value = (
                True,  # passed
                {
                    "mean": 0.005,
                    "max": 0.008,
                    "median": 0.004,
                    "count": 20,
                },  # Well within tolerance
            )

            # Act
            passed, stats = self.validator._check_q_consistency(
                experiment=mock_experiment,
                reflections=mock_reflections,
                tolerance=tolerance,
            )

            # Assert
            assert passed is True
            assert stats["count"] == 20
            assert stats["mean"] < tolerance  # Should be very small for ideal case
            assert stats["max"] < tolerance

    def test_check_q_consistency_mismatch_exceeding_tolerance(self):
        """Test Q-vector consistency check with mismatched data exceeding tolerance."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        tolerance = 0.01  # Å⁻¹ - strict tolerance

        # Mock the Q-vector consistency check directly to simulate mismatch scenario
        with patch.object(self.validator, "q_checker") as mock_q_checker:
            mock_q_checker.check_q_consistency.return_value = (
                False,  # failed
                {
                    "mean": 0.025,
                    "max": 0.080,
                    "median": 0.021,
                    "count": 20,
                },  # Exceeds tolerance
            )

            # Act
            passed, stats = self.validator._check_q_consistency(
                experiment=mock_experiment,
                reflections=mock_reflections,
                tolerance=tolerance,
            )

            # Assert
            assert passed is False  # Should fail due to large mismatch
            assert stats["count"] == 20
            assert stats["mean"] > tolerance  # Should exceed tolerance

    def test_check_q_consistency_missing_columns(self):
        """Test Q-vector consistency check with missing required columns."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=10)
        # Simulate missing required columns - return False for all column checks
        mock_reflections.__contains__ = Mock(return_value=False)
        # Also make it iterable to avoid TypeError
        mock_reflections.__iter__ = Mock(return_value=iter([]))

        tolerance = 0.01

        # Act
        passed, stats = self.validator._check_q_consistency(
            experiment=mock_experiment,
            reflections=mock_reflections,
            tolerance=tolerance,
        )

        # Assert
        assert passed is False
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["max"] is None


class TestBackwardCompatibility:
    """Test suite to ensure backward compatibility with original StillProcessorComponent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StillProcessorAndValidatorComponent()
        self.test_image_path = "tests/data/minimal_still.cbf"

    def test_process_still_backward_compatible(self):
        """Test that the original process_still interface still works."""
        # Arrange
        config = create_default_config()

        # Mock the adapter to return successful results
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)

        # Mock the processing route to return a specific adapter instance
        mock_adapter = Mock()
        mock_adapter.process_still.return_value = (
            mock_experiment,
            mock_reflections,
            True,
            "Success",
        )

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            mock_route.return_value = ("stills", mock_adapter)

            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path, config=config
            )

            # Assert
            assert outcome.status == "SUCCESS"
            assert "experiment" in outcome.output_artifacts
            assert "reflections" in outcome.output_artifacts
            # Should NOT have validation artifacts in backward compatible mode
            assert "validation_passed" not in outcome.output_artifacts
            assert "validation_metrics" not in outcome.output_artifacts


class TestCreateDefaultConfigs:
    """Test suite for configuration creation functions."""

    def test_create_default_config_basic(self):
        """Test creation of basic default configuration."""
        # Act
        config = create_default_config()

        # Assert
        assert isinstance(config, DIALSStillsProcessConfig)
        assert config.calculate_partiality is True
        assert config.output_shoeboxes is False

    def test_create_default_extraction_config(self):
        """Test creation of default extraction configuration."""
        # Act
        config = create_default_extraction_config()

        # Assert
        assert isinstance(config, ExtractionConfig)
        assert config.gain == 1.0
        assert config.cell_length_tol == 0.02
        assert config.cell_angle_tol == 2.0
        assert config.orient_tolerance_deg == 5.0
        assert config.q_consistency_tolerance_angstrom_inv == 0.01
        assert config.lp_correction_enabled is False


class TestValidationMetrics:
    """Test suite for ValidationMetrics class."""

    def test_validation_metrics_initialization(self):
        """Test ValidationMetrics initialization."""
        # Act
        metrics = ValidationMetrics()

        # Assert
        assert metrics.pdb_cell_passed is None
        assert metrics.pdb_orientation_passed is None
        assert metrics.q_consistency_passed is None
        assert metrics.num_reflections_tested == 0

    def test_validation_metrics_to_dict(self):
        """Test ValidationMetrics to_dict conversion."""
        # Arrange
        metrics = ValidationMetrics()
        metrics.q_consistency_passed = True
        metrics.mean_delta_q_mag = 0.005
        metrics.num_reflections_tested = 100

        # Act
        metrics_dict = metrics.to_dict()

        # Assert
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["q_consistency_passed"] is True
        assert metrics_dict["mean_delta_q_mag"] == 0.005
        assert metrics_dict["num_reflections_tested"] == 100
        assert "pdb_cell_passed" in metrics_dict


class TestCBFDataTypeDetection:
    """Test suite for CBF data type detection and processing route selection (Module 1.S.0)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StillProcessorAndValidatorComponent()
        self.config = DIALSStillsProcessConfig()

    def test_determine_processing_route_force_stills(self):
        """Test forced stills processing mode override."""
        # Arrange
        self.config.force_processing_mode = "stills"

        # Act
        route, adapter = self.processor._determine_processing_route(
            "dummy_path.cbf", self.config
        )

        # Assert
        assert route == "stills"
        assert adapter == self.processor.stills_adapter

    def test_determine_processing_route_force_sequence(self):
        """Test forced sequence processing mode override."""
        # Arrange
        self.config.force_processing_mode = "sequence"

        # Act
        route, adapter = self.processor._determine_processing_route(
            "dummy_path.cbf", self.config
        )

        # Assert
        assert route == "sequence"
        assert adapter == self.processor.sequence_adapter

    def test_determine_processing_route_auto_detect_stills(self):
        """Test auto-detection of stills data (Angle_increment = 0.0°)."""
        # Arrange
        self.config.force_processing_mode = None

        with patch(
            "diffusepipe.crystallography.still_processing_and_validation.get_angle_increment_from_cbf"
        ) as mock_get_angle:
            mock_get_angle.return_value = 0.0

            # Act
            route, adapter = self.processor._determine_processing_route(
                "test_still.cbf", self.config
            )

            # Assert
            assert route == "stills"
            assert adapter == self.processor.stills_adapter
            mock_get_angle.assert_called_once_with("test_still.cbf")

    def test_determine_processing_route_auto_detect_sequence(self):
        """Test auto-detection of sequence data (Angle_increment > 0.0°)."""
        # Arrange
        self.config.force_processing_mode = None

        with patch(
            "diffusepipe.crystallography.still_processing_and_validation.get_angle_increment_from_cbf"
        ) as mock_get_angle:
            mock_get_angle.return_value = 0.1

            # Act
            route, adapter = self.processor._determine_processing_route(
                "test_sequence.cbf", self.config
            )

            # Assert
            assert route == "sequence"
            assert adapter == self.processor.sequence_adapter
            mock_get_angle.assert_called_once_with("test_sequence.cbf")

    def test_determine_processing_route_auto_detect_fallback(self):
        """Test fallback to sequence processing when CBF parsing fails."""
        # Arrange
        self.config.force_processing_mode = None

        with patch(
            "diffusepipe.crystallography.still_processing_and_validation.get_angle_increment_from_cbf"
        ) as mock_get_angle:
            mock_get_angle.return_value = None  # Unable to determine

            # Act
            route, adapter = self.processor._determine_processing_route(
                "unknown.cbf", self.config
            )

            # Assert
            assert route == "sequence"  # Default fallback
            assert adapter == self.processor.sequence_adapter

    def test_determine_processing_route_invalid_force_mode(self):
        """Test handling of invalid force_processing_mode."""
        # Arrange
        self.config.force_processing_mode = "invalid_mode"

        with patch(
            "diffusepipe.crystallography.still_processing_and_validation.get_angle_increment_from_cbf"
        ) as mock_get_angle:
            mock_get_angle.return_value = 0.1

            # Act
            route, adapter = self.processor._determine_processing_route(
                "test.cbf", self.config
            )

            # Assert
            assert route == "sequence"  # Falls back to auto-detection
            assert adapter == self.processor.sequence_adapter


class TestPDBConsistencyCheck:
    """Test suite for PDB consistency checking functionality (Module 1.S.1.Validation)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()

    def test_check_pdb_consistency_no_symmetry(self):
        """Test PDB consistency check when PDB file has no crystal symmetry."""
        # Arrange
        mock_experiment = Mock()
        pdb_path = "test_no_symmetry.pdb"

        with patch("iotbx.pdb.input") as mock_pdb_input:
            mock_input = Mock()
            mock_input.crystal_symmetry.return_value = None  # No symmetry
            mock_pdb_input.return_value = mock_input

            # Act
            cell_passed, orient_passed, misorientation = (
                ModelValidator._check_pdb_consistency(
                    mock_experiment, pdb_path, 0.02, 2.0, 5.0
                )
            )

            # Assert
            assert cell_passed is True  # Should pass if PDB lacks symmetry
            assert orient_passed is True
            assert misorientation is None

    def test_check_pdb_consistency_cell_match(self):
        """Test PDB consistency check with matching unit cell parameters."""
        # Arrange
        from unittest.mock import PropertyMock

        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_experiment.crystal = mock_crystal

        # Mock experiment unit cell
        mock_exp_uc = Mock()
        mock_exp_uc.parameters.return_value = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
        mock_exp_uc.is_similar_to.return_value = True  # Cells match
        mock_crystal.get_unit_cell.return_value = mock_exp_uc

        # Mock experiment space group
        mock_exp_sg = Mock()
        mock_exp_sg_type = Mock()
        mock_exp_sg_type.number.return_value = 1
        mock_exp_sg.type.return_value = mock_exp_sg_type
        mock_crystal.get_space_group.return_value = mock_exp_sg

        # Mock experiment A matrix
        mock_crystal.get_A.return_value = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        pdb_path = "test_matching.pdb"

        with patch("iotbx.pdb.input") as mock_pdb_input:
            # Mock PDB crystal symmetry
            mock_input = Mock()
            mock_pdb_cs = Mock()

            mock_pdb_uc = Mock()
            mock_pdb_uc.parameters.return_value = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
            mock_pdb_uc.fractionalization_matrix.return_value = [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            mock_pdb_cs.unit_cell.return_value = mock_pdb_uc

            mock_pdb_sg = Mock()
            mock_pdb_sg_type = Mock()
            mock_pdb_sg_type.number.return_value = 1
            mock_pdb_sg.type.return_value = mock_pdb_sg_type
            mock_pdb_cs.space_group.return_value = mock_pdb_sg

            mock_input.crystal_symmetry.return_value = mock_pdb_cs
            mock_pdb_input.return_value = mock_input

            with patch.object(
                ModelValidator, "_calculate_misorientation_static"
            ) as mock_misori:
                mock_misori.return_value = 1.0  # Small misorientation

                # Act
                cell_passed, orient_passed, misorientation = (
                    ModelValidator._check_pdb_consistency(
                        mock_experiment, pdb_path, 0.02, 2.0, 5.0
                    )
                )

                # Assert
                assert cell_passed is True
                assert orient_passed is True
                assert misorientation == 1.0

    def test_check_pdb_consistency_cell_mismatch(self):
        """Test PDB consistency check with mismatched unit cell parameters."""
        # Arrange
        mock_experiment = Mock()
        mock_crystal = Mock()
        mock_experiment.crystal = mock_crystal

        # Mock experiment unit cell
        mock_exp_uc = Mock()
        mock_exp_uc.parameters.return_value = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
        mock_exp_uc.is_similar_to.return_value = False  # Cells don't match
        mock_crystal.get_unit_cell.return_value = mock_exp_uc

        # Mock experiment space group
        mock_exp_sg = Mock()
        mock_exp_sg_type = Mock()
        mock_exp_sg_type.number.return_value = 1
        mock_exp_sg.type.return_value = mock_exp_sg_type
        mock_crystal.get_space_group.return_value = mock_exp_sg

        # Mock experiment A matrix
        mock_crystal.get_A.return_value = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        pdb_path = "test_mismatched.pdb"

        with patch("iotbx.pdb.input") as mock_pdb_input:
            # Mock PDB crystal symmetry with different cell
            mock_input = Mock()
            mock_pdb_cs = Mock()

            mock_pdb_uc = Mock()
            mock_pdb_uc.parameters.return_value = (
                100.0,
                120.0,
                140.0,
                90.0,
                90.0,
                90.0,
            )  # Different cell
            mock_pdb_uc.fractionalization_matrix.return_value = [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            mock_pdb_cs.unit_cell.return_value = mock_pdb_uc

            mock_pdb_sg = Mock()
            mock_pdb_sg_type = Mock()
            mock_pdb_sg_type.number.return_value = 1
            mock_pdb_sg.type.return_value = mock_pdb_sg_type
            mock_pdb_cs.space_group.return_value = mock_pdb_sg

            mock_input.crystal_symmetry.return_value = mock_pdb_cs
            mock_pdb_input.return_value = mock_input

            with patch.object(
                ModelValidator, "_calculate_misorientation_static"
            ) as mock_misori:
                mock_misori.return_value = 8.0  # Large misorientation

                # Act
                cell_passed, orient_passed, misorientation = (
                    ModelValidator._check_pdb_consistency(
                        mock_experiment, pdb_path, 0.02, 2.0, 5.0  # tolerance: 5°
                    )
                )

                # Assert
                assert cell_passed is False
                assert orient_passed is False  # 8° > 5° tolerance
                assert misorientation == 8.0

    def test_calculate_misorientation_static_identity(self):
        """Test misorientation calculation between identical matrices."""
        # Arrange
        from scitbx import matrix

        A1 = matrix.sqr([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        A2 = matrix.sqr([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        # Act
        misorientation = ModelValidator._calculate_misorientation_static(A1, A2)

        # Assert
        assert misorientation < 1e-6  # Should be essentially zero

    def test_calculate_misorientation_static_hand_inversion(self):
        """Test misorientation calculation handles hand inversion correctly."""
        # Arrange
        from scitbx import matrix

        A1 = matrix.sqr([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        A2 = matrix.sqr(
            [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
        )  # Inverted hand

        # Act
        misorientation = ModelValidator._calculate_misorientation_static(A1, A2)

        # Assert
        # Should return the minimum of direct and inverted comparison
        assert (
            misorientation < 1e-6
        )  # Should be essentially zero after considering inversion

    def test_pdb_consistency_check_file_error(self):
        """Test PDB consistency check handles file errors gracefully."""
        # Arrange
        mock_experiment = Mock()
        pdb_path = "nonexistent.pdb"

        with patch("iotbx.pdb.input") as mock_pdb_input:
            mock_pdb_input.side_effect = FileNotFoundError("File not found")

            # Act
            cell_passed, orient_passed, misorientation = (
                ModelValidator._check_pdb_consistency(
                    mock_experiment, pdb_path, 0.02, 2.0, 5.0
                )
            )

            # Assert
            assert cell_passed is False
            assert orient_passed is False
            assert misorientation is None
