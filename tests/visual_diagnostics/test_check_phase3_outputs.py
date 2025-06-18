"""
Tests for Phase 3 visual diagnostics script.

This module tests the check_phase3_outputs.py script functionality using
synthetic Phase 3 output data to verify plot generation and summary creation.
"""

import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys

# Add the scripts directory to the path for importing the script
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent / "scripts" / "visual_diagnostics"
    ),
)

try:
    from check_phase3_outputs import (
        load_grid_definition,
        load_scaling_parameters,
        load_voxel_data,
        generate_grid_summary,
        generate_voxel_occupancy_plots,
        generate_scaling_parameter_plots,
        generate_merged_voxel_plots,
        generate_comprehensive_summary,
    )
except ImportError as e:
    pytest.skip(
        f"Could not import check_phase3_outputs module: {e}", allow_module_level=True
    )


@pytest.fixture
def synthetic_grid_definition():
    """Create synthetic grid definition data."""
    return {
        "crystal_avg_ref": {
            "unit_cell_params": [78.0, 78.0, 37.0, 90.0, 90.0, 90.0],
            "space_group": "P43212",
        },
        "hkl_bounds": {
            "h_min": -50,
            "h_max": 50,
            "k_min": -50,
            "k_max": 50,
            "l_min": -25,
            "l_max": 25,
        },
        "ndiv_h": 100,
        "ndiv_k": 100,
        "ndiv_l": 50,
        "total_voxels": 500000,
    }


@pytest.fixture
def synthetic_scaling_parameters():
    """Create synthetic scaling parameters data."""
    return {
        "refined_parameters": {
            "still_0": {"multiplicative_scale": 1.05, "additive_offset": 0.0},
            "still_1": {"multiplicative_scale": 0.95, "additive_offset": 0.0},
            "still_2": {"multiplicative_scale": 1.02, "additive_offset": 0.0},
        },
        "refinement_statistics": {
            "n_iterations": 5,
            "final_r_factor": 0.15,
            "convergence_achieved": True,
            "parameter_shifts": {"multiplicative_scale": 0.001},
        },
        "resolution_smoother": {"enabled": False, "control_points": []},
    }


@pytest.fixture
def synthetic_voxel_data():
    """Create synthetic voxel data."""
    n_voxels = 1000
    np.random.seed(42)  # For reproducible tests

    return {
        "voxel_indices": np.arange(n_voxels),
        "H_center": np.random.randint(-50, 51, n_voxels),
        "K_center": np.random.randint(-50, 51, n_voxels),
        "L_center": np.random.randint(-25, 26, n_voxels),
        "q_center_x": np.random.uniform(-2, 2, n_voxels),
        "q_center_y": np.random.uniform(-2, 2, n_voxels),
        "q_center_z": np.random.uniform(-1, 1, n_voxels),
        "q_magnitude_center": np.random.uniform(0.1, 2.5, n_voxels),
        "I_merged_relative": np.random.exponential(100, n_voxels),
        "Sigma_merged_relative": np.random.exponential(10, n_voxels),
        "num_observations": np.random.poisson(5, n_voxels) + 1,
    }


class TestDataLoading:
    """Test data loading functions."""

    def test_load_grid_definition(self, synthetic_grid_definition):
        """Test loading grid definition from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(synthetic_grid_definition, f)
            temp_file = f.name

        try:
            loaded_data = load_grid_definition(temp_file)
            assert loaded_data == synthetic_grid_definition
            assert "crystal_avg_ref" in loaded_data
            assert "hkl_bounds" in loaded_data
            assert "total_voxels" in loaded_data
        finally:
            Path(temp_file).unlink()

    def test_load_grid_definition_missing_key(self):
        """Test loading grid definition with missing required key."""
        incomplete_data = {"crystal_avg_ref": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(incomplete_data, f)
            temp_file = f.name

        try:
            with pytest.raises(KeyError, match="Required key.*not found"):
                load_grid_definition(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_scaling_parameters(self, synthetic_scaling_parameters):
        """Test loading scaling parameters from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(synthetic_scaling_parameters, f)
            temp_file = f.name

        try:
            loaded_data = load_scaling_parameters(temp_file)
            assert loaded_data == synthetic_scaling_parameters
            assert "refined_parameters" in loaded_data
            assert "refinement_statistics" in loaded_data
        finally:
            Path(temp_file).unlink()

    def test_load_voxel_data_npz(self, synthetic_voxel_data):
        """Test loading voxel data from NPZ file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **synthetic_voxel_data)
            temp_file = f.name

        try:
            loaded_data = load_voxel_data(temp_file)

            # Check all required keys are present
            for key in synthetic_voxel_data.keys():
                assert key in loaded_data
                np.testing.assert_array_equal(
                    loaded_data[key], synthetic_voxel_data[key]
                )
        finally:
            Path(temp_file).unlink()

    def test_load_voxel_data_missing_key(self):
        """Test loading voxel data with missing required key."""
        incomplete_data = {"voxel_indices": np.array([1, 2, 3])}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **incomplete_data)
            temp_file = f.name

        try:
            with pytest.raises(KeyError, match="Required key.*not found"):
                load_voxel_data(temp_file)
        finally:
            Path(temp_file).unlink()


class TestPlotGeneration:
    """Test plot generation functions."""

    @patch("matplotlib.figure.Figure.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_grid_summary(
        self, mock_close, mock_savefig, synthetic_grid_definition
    ):
        """Test grid summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            generate_grid_summary(synthetic_grid_definition, output_dir)

            # Check that text summary was created
            summary_file = output_dir / "grid_summary.txt"
            assert summary_file.exists()

            # Verify content
            content = summary_file.read_text()
            assert "Global Voxel Grid Summary" in content
            assert "78.0" in content  # Unit cell parameter
            assert "P43212" in content  # Space group
            assert "500,000" in content  # Total voxels

            # Check that plot was attempted to be saved
            mock_savefig.assert_called()
            mock_close.assert_called()

    @patch("matplotlib.figure.Figure.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_voxel_occupancy_plots(
        self, mock_close, mock_savefig, synthetic_voxel_data, synthetic_grid_definition
    ):
        """Test voxel occupancy plots generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            stats = generate_voxel_occupancy_plots(
                synthetic_voxel_data, synthetic_grid_definition, output_dir
            )

            # Check returned statistics
            assert isinstance(stats, dict)
            assert "min_observations" in stats
            assert "max_observations" in stats
            assert "mean_observations" in stats
            assert "total_voxels" in stats

            # Verify statistics are reasonable
            assert stats["min_observations"] >= 0
            assert stats["max_observations"] >= stats["min_observations"]
            assert stats["total_voxels"] == len(
                synthetic_voxel_data["num_observations"]
            )

            # Check that plots were attempted to be saved
            assert mock_savefig.call_count >= 3  # At least slice plots + histogram

    @patch("matplotlib.figure.Figure.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_scaling_parameter_plots(
        self, mock_close, mock_savefig, synthetic_scaling_parameters
    ):
        """Test scaling parameter plots generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            generate_scaling_parameter_plots(synthetic_scaling_parameters, output_dir)

            # Check that summary file was created
            summary_file = output_dir / "scaling_parameters_summary.txt"
            assert summary_file.exists()

            # Verify content
            content = summary_file.read_text()
            assert "Scaling Model Parameters Summary" in content
            assert "Number of Stills: 3" in content

            # Check that plots were attempted to be saved
            mock_savefig.assert_called()

    @patch("matplotlib.figure.Figure.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_merged_voxel_plots(
        self, mock_close, mock_savefig, synthetic_voxel_data, synthetic_grid_definition
    ):
        """Test merged voxel data plots generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            max_plot_points = 500

            generate_merged_voxel_plots(
                synthetic_voxel_data,
                synthetic_grid_definition,
                output_dir,
                max_plot_points,
            )

            # Check that multiple plots were attempted to be saved
            # Should include: intensity slices, sigma slices, I/sigma slices,
            # radial average, intensity histogram
            assert mock_savefig.call_count >= 8


class TestSummaryGeneration:
    """Test summary generation functions."""

    def test_generate_comprehensive_summary(
        self,
        synthetic_grid_definition,
        synthetic_scaling_parameters,
        synthetic_voxel_data,
    ):
        """Test comprehensive summary generation."""
        # Test uses real datetime module, which we've already imported

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create synthetic occupancy stats
            occupancy_stats = {
                "min_observations": 1,
                "max_observations": 15,
                "mean_observations": 5.2,
                "median_observations": 5.0,
                "total_observations": 5200,
                "voxels_with_data": 1000,
                "total_voxels": 1000,
                "percent_voxels_lt_3": 15.0,
            }

            input_files = {
                "Grid Definition": "test_grid.json",
                "Scaling Parameters": "test_scaling.json",
                "Voxel Data": "test_voxel.npz",
            }

            generate_comprehensive_summary(
                synthetic_grid_definition,
                synthetic_scaling_parameters,
                synthetic_voxel_data,
                occupancy_stats,
                input_files,
                output_dir,
            )

            # Check that summary file was created
            summary_file = output_dir / "phase3_diagnostics_summary.txt"
            assert summary_file.exists()

            # Verify content
            content = summary_file.read_text()
            assert "Phase 3 Diagnostics Summary Report" in content
            assert "Generated on:" in content
            assert "Global Voxel Grid:" in content
            assert "Voxel Occupancy:" in content
            assert "Relative Scaling:" in content
            assert "Merged Intensities:" in content
            assert "Resolution Coverage:" in content
            assert "Generated Diagnostic Plots:" in content


class TestErrorHandling:
    """Test error handling in diagnostic functions."""

    def test_load_grid_definition_file_not_found(self):
        """Test loading grid definition with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_grid_definition("nonexistent_file.json")

    def test_load_voxel_data_unsupported_format(self):
        """Test loading voxel data with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"dummy content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="must be .npz or .hdf5"):
                load_voxel_data(temp_file)
        finally:
            Path(temp_file).unlink()


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflow."""

    @patch("matplotlib.figure.Figure.savefig")
    @patch("matplotlib.pyplot.close")
    def test_full_diagnostic_workflow(
        self,
        mock_close,
        mock_savefig,
        synthetic_grid_definition,
        synthetic_scaling_parameters,
        synthetic_voxel_data,
    ):
        """Test the complete diagnostic workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input files
            grid_file = temp_path / "grid_def.json"
            scaling_file = temp_path / "scaling_params.json"
            voxel_file = temp_path / "voxel_data.npz"
            output_dir = temp_path / "diagnostics"

            with open(grid_file, "w") as f:
                json.dump(synthetic_grid_definition, f)

            with open(scaling_file, "w") as f:
                json.dump(synthetic_scaling_parameters, f)

            np.savez_compressed(voxel_file, **synthetic_voxel_data)

            output_dir.mkdir()

            # Load data
            grid_def = load_grid_definition(str(grid_file))
            scaling_params = load_scaling_parameters(str(scaling_file))
            voxel_data = load_voxel_data(str(voxel_file))

            # Generate all diagnostics
            generate_grid_summary(grid_def, output_dir)
            occupancy_stats = generate_voxel_occupancy_plots(
                voxel_data, grid_def, output_dir
            )
            generate_scaling_parameter_plots(scaling_params, output_dir)
            generate_merged_voxel_plots(voxel_data, grid_def, output_dir, 500)

            # Verify expected files were created
            expected_files = ["grid_summary.txt", "scaling_parameters_summary.txt"]

            for filename in expected_files:
                assert (output_dir / filename).exists(), f"{filename} was not created"

            # Verify plots were attempted to be saved
            assert mock_savefig.call_count > 10  # Multiple plots should be generated
