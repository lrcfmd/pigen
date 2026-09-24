import pandas as pd
from chgnet.model.model import CHGNet
chgnet = CHGNet.load()
from pymatgen.core import Structure
import sys


def chgnet_sp(df):
    energies = []
    for cif in df['cif']:
        struct = Structure.from_str(cif, fmt='cif')
        prediction = chgnet.predict_structure(struct)
        energy = prediction['energy'[0]] #(eV/atom)
        print(energy, 'eV/atom')
        energies.append(energy)
    df['chgnet_energy'] = energies
    return df

dfname = sys.argv[1]
df = pd.read_csv(dfname)
df = chgnet_sp(df)

