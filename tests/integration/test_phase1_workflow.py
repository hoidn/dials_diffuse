"""
Integration test for complete Phase 1 workflow.

This test demonstrates the complete Phase 1 pipeline from still processing
through mask generation, validating the integration between all components.
"""

from unittest.mock import Mock, patch, MagicMock

from diffusepipe.crystallography.still_processing_and_validation import (
    StillProcessorAndValidatorComponent,
    create_default_config,
    create_default_extraction_config,
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


class TestPhase1Workflow:
    """Integration test for complete Phase 1 workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StillProcessorAndValidatorComponent()
        self.pixel_generator = PixelMaskGenerator()
        self.bragg_generator = BraggMaskGenerator()

        self.test_image_path = "tests/data/test_still.cbf"
        self.test_pdb_path = "tests/data/reference.pdb"

    def test_complete_phase1_workflow_success(self):
        """Test complete Phase 1 workflow with successful processing and validation."""
        # Arrange
        dials_config = create_default_config()
        extraction_config = create_default_extraction_config()
        static_params = create_default_static_params()
        dynamic_params = create_default_dynamic_params()
        bragg_config = create_default_bragg_mask_config()

        # Mock DIALS objects
        mock_experiment = Mock()
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (100, 50)

        mock_detector.__iter__ = MagicMock(side_effect=lambda: iter([mock_panel]))
        mock_detector.__len__ = Mock(return_value=1)
        mock_experiment.detector = mock_detector

        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)
        mock_reflections.__len__ = Mock(return_value=100)

        # Mock image sets for pixel mask generation
        mock_image_set = Mock()
        # Create a real numpy array for panel data that can be processed
        import numpy as np

        mock_panel_data_instance = (
            np.ones((50, 100), dtype=np.uint16) * 1000
        )  # Realistic detector data
        mock_image_set.get_raw_data.return_value = (
            mock_panel_data_instance,
        )  # Return tuple containing one panel
        representative_images = [mock_image_set]

        # Remove flex mocking to allow real flex operations for internal workings
        # The test will work with real flex arrays but mock the external DIALS calls

        # Mock DIALS processing and validation
        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(
                self.processor.stills_adapter, "process_still"
            ) as mock_process:
                with patch.object(
                    self.processor.validator, "validate_geometry"
                ) as mock_validate:
                    mock_route.return_value = ("stills", self.processor.stills_adapter)
                    mock_process.return_value = (
                        mock_experiment,
                        mock_reflections,
                        True,
                        "Success",
                    )
                    mock_validate.return_value = (True, Mock())

                    # Mock DIALS generate mask
                    with patch.object(
                        self.bragg_generator.dials_adapter, "generate_bragg_mask"
                    ) as mock_gen_mask:
                        # Create a real flex array for the bragg mask
                        from dials.array_family import flex

                        mock_bragg_mask = flex.bool(
                            flex.grid(50, 100), True
                        )  # Real flex array
                        mock_gen_mask.return_value = (
                            (mock_bragg_mask,),
                            True,
                            "Mask generated",
                        )

                        # Act - Step 1: Process and validate still
                        still_outcome = self.processor.process_and_validate_still(
                            image_path=self.test_image_path,
                            config=dials_config,
                            extraction_config=extraction_config,
                            external_pdb_path=self.test_pdb_path,
                        )

                        # Assert Step 1
                        assert still_outcome.status == "SUCCESS"
                        assert (
                            still_outcome.output_artifacts["validation_passed"] is True
                        )

                        # Act - Step 2: Generate global pixel mask
                        global_pixel_mask = (
                            self.pixel_generator.generate_combined_pixel_mask(
                                detector=mock_detector,
                                static_params=static_params,
                                representative_images=representative_images,
                                dynamic_params=dynamic_params,
                            )
                        )

                        # Assert Step 2
                        assert isinstance(global_pixel_mask, tuple)
                        assert len(global_pixel_mask) == 1  # One panel

                        # Act - Step 3: Generate Bragg mask for this still
                        bragg_mask = (
                            self.bragg_generator.generate_bragg_mask_from_spots(
                                experiment=mock_experiment,
                                reflections=mock_reflections,
                                config=bragg_config,
                            )
                        )

                        # Assert Step 3
                        assert isinstance(bragg_mask, tuple)
                        assert len(bragg_mask) == 1  # One panel

                        # Act - Step 4: Generate total mask
                        total_mask = self.bragg_generator.get_total_mask_for_still(
                            bragg_mask=bragg_mask,
                            global_pixel_mask=global_pixel_mask,
                        )

                        # Assert Step 4
                        assert isinstance(total_mask, tuple)
                        assert len(total_mask) == 1  # One panel

                        # Verify all components were called
                        mock_process.assert_called_once()
                        mock_validate.assert_called_once()
                        mock_gen_mask.assert_called_once()

    def test_phase1_workflow_with_validation_failure(self):
        """Test Phase 1 workflow when geometric validation fails."""
        # Arrange
        dials_config = create_default_config()
        extraction_config = create_default_extraction_config()

        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)

        # Mock DIALS processing to succeed but validation to fail
        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(
                self.processor.stills_adapter, "process_still"
            ) as mock_process:
                with patch.object(
                    self.processor.validator, "validate_geometry"
                ) as mock_validate:
                    mock_route.return_value = ("stills", self.processor.stills_adapter)
                    mock_process.return_value = (
                        mock_experiment,
                        mock_reflections,
                        True,
                        "Success",
                    )
                    mock_validate.return_value = (False, Mock())  # Validation fails

                    # Act
                    outcome = self.processor.process_and_validate_still(
                        image_path=self.test_image_path,
                        config=dials_config,
                        extraction_config=extraction_config,
                    )

                    # Assert
                    assert outcome.status == "FAILURE_GEOMETRY_VALIDATION"
                    assert outcome.output_artifacts["validation_passed"] is False

                # Subsequent mask generation should be skipped in real orchestrator
                # This test demonstrates that the validation caught the issue

    def test_phase1_workflow_with_dials_failure(self):
        """Test Phase 1 workflow when DIALS processing fails."""
        # Arrange
        dials_config = create_default_config()
        extraction_config = create_default_extraction_config()

        # Mock DIALS processing to fail
        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(
                self.processor.stills_adapter, "process_still"
            ) as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.return_value = (None, None, False, "DIALS failed")

                # Act
                outcome = self.processor.process_and_validate_still(
                    image_path=self.test_image_path,
                    config=dials_config,
                    extraction_config=extraction_config,
                )

                # Assert
                assert outcome.status == "FAILURE_DIALS_PROCESSING"

            # Validation and mask generation should be skipped
            # This test demonstrates proper error handling in the pipeline

    def test_phase1_mask_compatibility_validation(self):
        """Test validation of mask compatibility between components."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (100, 50)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))
        mock_detector.__len__ = Mock(return_value=1)

        # Mock masks with compatible dimensions
        with patch("diffusepipe.masking.bragg_mask_generator.flex"):
            mock_pixel_mask = Mock()
            mock_pixel_mask.__len__ = Mock(return_value=5000)
            mock_bragg_mask = Mock()
            mock_bragg_mask.__len__ = Mock(return_value=5000)

            pixel_mask_tuple = (mock_pixel_mask,)
            bragg_mask_tuple = (mock_bragg_mask,)

            # Act
            from diffusepipe.masking.bragg_mask_generator import (
                validate_mask_compatibility,
            )

            is_compatible = validate_mask_compatibility(
                bragg_mask_tuple, pixel_mask_tuple
            )

            # Assert
            assert is_compatible is True

    def test_phase1_component_integration_interfaces(self):
        """Test that all Phase 1 components have compatible interfaces."""
        # This test validates that the components can be connected together
        # without runtime interface mismatches

        # Test StillProcessorAndValidatorComponent interface
        processor = StillProcessorAndValidatorComponent()
        assert hasattr(processor, "process_and_validate_still")
        assert hasattr(processor, "stills_adapter")
        assert hasattr(processor, "sequence_adapter")
        assert hasattr(processor, "validator")

        # Test PixelMaskGenerator interface
        pixel_gen = PixelMaskGenerator()
        assert hasattr(pixel_gen, "generate_combined_pixel_mask")
        assert hasattr(pixel_gen, "generate_static_mask")
        assert hasattr(pixel_gen, "generate_dynamic_mask")

        # Test BraggMaskGenerator interface
        bragg_gen = BraggMaskGenerator()
        assert hasattr(bragg_gen, "generate_bragg_mask_from_spots")
        assert hasattr(bragg_gen, "generate_bragg_mask_from_shoeboxes")
        assert hasattr(bragg_gen, "get_total_mask_for_still")

        # Test configuration creation functions
        dials_config = create_default_config()
        extraction_config = create_default_extraction_config()
        static_params = create_default_static_params()
        dynamic_params = create_default_dynamic_params()
        bragg_config = create_default_bragg_mask_config()

        # All configurations should be created without errors
        assert dials_config is not None
        assert extraction_config is not None
        assert static_params is not None
        assert dynamic_params is not None
        assert bragg_config is not None
