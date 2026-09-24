from __future__ import annotations

import warnings

from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

import sys
import numpy as np

def safe_atomic_numbers(cif) -> np.ndarray | None:
    """Return Z array, or None if the structure would raise."""
    crystal = Structure.from_str(cif, fmt='cif')
    if not crystal.is_ordered:
        return None
    return np.array(crystal.atomic_numbers)

import pandas as pd

df = pd.read_csv(sys.argv[1])

df['safe_numbers'] = df['cif'].apply(safe_atomic_numbers)

print(df.shape)
bf = df[df['safe_numbers'].notna()]
print(bf.shape)

bf.to_csv(f'BVSE_computed_chunks_with_psi_and_MP_{bf.shape[0]}.csv')
