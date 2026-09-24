import matplotlib.pyplot as plt
from ase.io import *
import json, pickle
import sys
import numpy as np
import pandas as pd
from io import StringIO
from tqdm import tqdm

# get -ln (gr)
#for k in df: 
#    df[k]['count'].apply(lambda x: -np.log(x))

def compute_spp(df: pd.DataFrame):
    """
    df is a DataFrame of strcutures
    """

    # Opening up precomputed SPPs
    with open('spps/SPP_collected.json', 'r') as f:
        SPP = json.load(f)
    errors = []
    print('--- Computing SPPs ---')
    for i, cif in enumerate(tqdm(df['strcuture'])):
        errors.append(pairs(cif, SPP))
    return errors
    

def newatoms(cif):
    psfile = StringIO(cif)
    atoms = read(psfile, format='cif')
    return atoms

def tobin(x, bins):
    i = np.digitize(x, bins)
    return bins[i]

def spp_error(x, spp):
    i = tobin(x, [float(i) for i in spp.keys()])
    return spp[str(i)]

def pairs(cif, SPP):
    try:
        atoms = newatoms(cif)
        pairs = []
        syms = atoms.get_chemical_symbols()
        distances = atoms.get_all_distances()
        U = 0
        N = 0
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                if j == i: continue
                pair = '-'.join(sorted([a,b]))
                pairs.append(pair)
                error = spp_error(distances[i][j], SPP[pair])
                #print(pair, error)
                U += error
                N += 1
        return round(U/N, 4)
    except Exception:
        return np.nan
    
# if __name__ == '__main__':
#     train = pd.read_csv('data/full_data/train.csv')

#     errors = compute_spp(train)
#     train['spps_error'] = errors
#     train.to_csv('data/full_data/train_spps.csv', index=False)

#     val = pd.read_csv('data/full_data/val.csv')
#     errors = compute_spp(val)
#     val['spps_error'] = errors
#     val.to_csv('data/full_data/val_spps.csv', index=False)

#     test = pd.read_csv('data/full_data/test.csv')
#     errors = compute_spp(test)
#     test['spps_error'] = errors
#     test.to_csv('data/full_data/test_spps.csv', index=False)
# if __name__=="__main__":
#     df = pd.read_csv("/home/fedeotto/crys_repaint/ckpts/coordiv_cmpt/structures/denovo/structures_[4, 0.7]_-1.0.csv")
#     df = df.dropna()

#     df = compute_spp(df)

#     df.to_csv("/home/fedeotto/crys_repaint/ckpts/coordiv_cmpt/structures/denovo/structures_[4, 0.7]_-1.0_spp.csv", index=False)