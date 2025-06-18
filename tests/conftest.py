"""Shared pytest fixtures and configuration for the test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_cbf_file(tmp_path):
    """Create a minimal mock CBF file for testing."""
    cbf_path = tmp_path / "test_image.cbf"
    # Create a minimal CBF file (this would need to be a valid CBF in real tests)
    cbf_path.write_text("# Mock CBF file for testing")
    return str(cbf_path)


@pytest.fixture
def mock_pdb_file(tmp_path):
    """Create a minimal mock PDB file for testing."""
    pdb_path = tmp_path / "test.pdb"
    # Create a minimal PDB file content
    pdb_content = """HEADER    TEST STRUCTURE
CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
ATOM      1  CA  ALA A   1      25.000  25.000  25.000  1.00 20.00           C
END
"""
    pdb_path.write_text(pdb_content)
    return str(pdb_path)
