import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure


def is_valid_cif(cif_str: str) -> bool:
    try:
        struct = Structure.from_str(str(cif_str), fmt='cif')
        return len(struct) > 0
    except Exception:
        return False

df = pd.read_csv('val.csv')
print(f"Before validation: {len(df)}")

mask = [is_valid_cif(c) for c in tqdm(df['cif'], desc='Validating CIFs')]
df_valid = df[mask].reset_index(drop=True)

print(f"After  validation: {len(df_valid)}")
print(f"Dropped: {(~pd.Series(mask)).sum()}")

df_valid.to_csv('val.csv', index=False)
