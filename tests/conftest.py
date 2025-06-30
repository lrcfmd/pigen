import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[1]

@pytest.fixture
def dummy_data_path(project_root):
    """Path to example data file used in tests."""
    return project_root / "data" / "Alex_MP_20_M_LED/test.csv"

def test_data_file_exists(dummy_data_path):
    assert dummy_data_path.exists(), "Expected dummy data file not found"

