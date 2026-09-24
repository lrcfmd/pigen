import warnings
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict

from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.cif import CifWriter


def try_order_structure(cif_str: str) -> tuple[str | None, str]:
    """
    Parse a CIF string, order if disordered, return (cif_string | None, status).

    Status values:
        ordered_original      — already ordered, passed through
        ordered_transformed   — disordered, successfully ordered via Ewald minimisation
        oxi_state_failed      — could not assign oxidation states (dropped)
        ordering_failed       — oxi states ok but transformation crashed (dropped)
        parse_failed          — CIF unparseable (dropped)
    """
    # ------------------------------------------------------------------ parse
    try:
        struct = Structure.from_str(cif_str, fmt='cif')
    except Exception as e:
        return None, f'parse_failed: {e}'

    # --------------------------------------------------------- already ordered
    if struct.is_ordered:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Site labels are not unique')
            return CifWriter(struct).__str__(), 'ordered_original'

    # ------------------------------------------------- assign oxidation states
    # Attempt 1: BVAnalyzer (best quality)
    struct_oxi = None
    try:
        bva = BVAnalyzer()
        struct_oxi = bva.get_oxi_state_decorated_structure(struct)
    except Exception:
        pass

    # Attempt 2: fallback — assign oxi=0 to all elements on a clean copy
    if struct_oxi is None:
        try:
            struct_clean = struct.copy()
            struct_clean.remove_oxidation_states()          # guarantee clean slate
            oxi_dict = {str(el): 0 for el in struct_clean.composition.elements}
            struct_clean.add_oxidation_state_by_element(oxi_dict)
            struct_oxi = struct_clean
        except Exception as e:
            return None, f'oxi_state_failed: {e}'

    # ------------------------------------------------------------------ order
    try:
        trans = OrderDisorderedStructureTransformation()
        ordered = trans.apply_transformation(struct_oxi)    # lowest-energy ordering
        ordered.remove_oxidation_states()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Site labels are not unique')
            return CifWriter(ordered).__str__(), 'ordered_transformed'
    except Exception as e:
        return None, f'ordering_failed: {e}'


def process_csv(
    input_path:  str,
    output_path: str,
    cif_col:     str = 'cif',
) -> pd.DataFrame:

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    cif_out, statuses = [], []

    for cif_str in tqdm(df[cif_col], desc='Processing structures'):
        cif_result, status = try_order_structure(str(cif_str))
        cif_out.append(cif_result)
        statuses.append(status)

    df[cif_col]          = cif_out
    df['ordering_status'] = statuses

    # --------------------------------------------------------- status summary
    counts = Counter(statuses)
    print("\nStatus breakdown:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:50s}: {v}")

    n_kept    = sum(v for k, v in counts.items() if 'failed' not in k)
    n_dropped = sum(v for k, v in counts.items() if 'failed'     in k)
    print(f"\n  Total kept   : {n_kept}")
    print(f"  Total dropped: {n_dropped}")

    # ------------------------------------------------------------ drop failed
    df = df[df[cif_col].notna()].reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Order disordered structures in a CSV')
    parser.add_argument('--input',   type=str, default='train.csv',         help='Input CSV path')
    parser.add_argument('--output',  type=str, default='train_ordered_full.csv', help='Output CSV path')
    parser.add_argument('--cif_col', type=str, default='cif',               help='Column name containing CIF strings')
    args = parser.parse_args()

    process_csv(args.input, args.output, args.cif_col)
