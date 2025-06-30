import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def example_df():
    this_dir = Path(__file__).parent
    db_path = this_dir / "fixtures" / "test_db.csv"  
    return pd.read_csv(db_path)

def test_read_csv(example_df):
    assert not example_df.empty

def test_structure_parsing(example_df):
    from pymatgen.core import Structure
    struct = Structure.from_str(example_df.loc[0, 'cif'], fmt="cif")
    assert isinstance(struct, Structure)

