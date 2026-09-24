import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.cif import CifWriter

def try_order_structure(cif_str: str) -> tuple[str | None, str]:
    """
    Parse CIF string, order if disordered, return (cif_string_or_None, status).
    Status: 'ordered_original' | 'ordered_transformed' | 'failed'
    """
    try:
        struct = Structure.from_str(cif_str, fmt='cif')
    except Exception as e:
        return None, f'parse_failed: {e}'

    if struct.is_ordered:
        return CifWriter(struct).__str__(), 'ordered_original'

    # Disordered — need to order it
    # OrderDisorderedStructureTransformation requires oxidation states for Ewald
    try:
        bva = BVAnalyzer()
        struct_oxi = bva.get_oxi_state_decorated_structure(struct)
    except Exception:
        # BVAnalyzer failed — try guessing oxi states as 0
        try:
            struct_oxi = struct.copy()
            struct_oxi.add_oxidation_state_by_element(
                {str(el): 0 for el in struct.composition.elements}
            )
        except Exception as e:
            return None, f'oxi_state_failed: {e}'

    try:
        trans = OrderDisorderedStructureTransformation()
        ordered = trans.apply_transformation(struct_oxi)   # returns lowest-energy ordering
        ordered.remove_oxidation_states()                  # clean up for downstream use
        return CifWriter(ordered).__str__(), 'ordered_transformed'
    except Exception as e:
        return None, f'ordering_failed: {e}'


def process_csv(input_path: str, output_path: str, cif_col: str = 'cif'):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    cif_out, statuses = [], []

    for cif_str in tqdm(df[cif_col], desc='Processing structures'):
        cif_result, status = try_order_structure(str(cif_str))
        cif_out.append(cif_result)
        statuses.append(status)

    df[cif_col] = cif_out
    df['ordering_status'] = statuses

    # Report
    from collections import Counter
    counts = Counter(statuses)
    print("\nStatus summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Drop failed rows
    n_before = len(df)
    df = df[df[cif_col].notna()].reset_index(drop=True)
    print(f"\nDropped {n_before - len(df)} failed rows, {len(df)} remaining")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


if __name__ == '__main__':
    df_clean = process_csv('val_disordered.csv', 'val.csv', cif_col='cif')
