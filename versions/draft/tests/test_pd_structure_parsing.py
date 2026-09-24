import pytest
import pandas as pd

@pytest.fixture
def example_df():
    return pd.read_csv("tests/fixtures/test_db.csv")

def test_read_csv(example_df):
    assert not example_df.empty

def test_structure_parsing(example_df):
    from pymatgen.core import Structure
    struct = Structure.from_str(example_df.loc[0, 'cif'], fmt="cif")
    assert isinstance(struct, Structure)

