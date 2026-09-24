import numpy as np
import sys
import pdd
import time
from pdd_scaled import PDD, PDD_scale, PDD_to_AMD, PDD_scale_split
import pandas as pd
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

def s2cif(s):
    try:
        s = str(CifWriter(s))
    except:
        s = None
    return s

def compute_periodic_sets(df, col='structure'):
    periodic_sets = []

    for structure in df[col]:
            structure = Structure.from_str(structure, fmt='cif')
            psets = pdd.read_cif_pmg(structure)
            periodic_sets.append(psets)

    return periodic_sets

def _compute_periodic_sets(df, col='structure'):
    periodic_sets = []
    could_not = 0

    for structure in df[col]:
        try:
            if structure is None:
                periodic_sets.append(None)
                continue

            psets = pdd.read_cif_pmg(structure)

            if psets is None:
                periodic_sets.append(None)
            else:
                periodic_sets.append(psets)

        except Exception:
            could_not += 1
            periodic_sets.append(None)

    print('Could not build periodic sets:', could_not, '/', df.shape[0])

    return periodic_sets

if __name__ == '__main__':
    k = 100
    dfile = sys.argv[1]
    df = pd.read_csv(dfile)
    
    #del df['structure']
    #df.to_csv('ConstrastiveLearning_StrSpace/test.csv', index=False)

    periodic_sets = compute_periodic_sets(df, col='cif') 
    print('Computing PDD Scaled, k=', k)
    #pdds = [PDD(crystal_1, k) if crystal_1 is not None else None for crystal_1 in periodic_sets] 
    pdds = [pdd.pdd(crystal_1, k) if crystal_1 is not None else None for crystal_1 in periodic_sets] 
    df['amd'] = [PDD_to_AMD(pdd_1) if pdd_1 is not None else None for pdd_1 in pdds]
    #pdds = [PDD_scale(crystal_1, k) if crystal_1 is not None else None for crystal_1 in periodic_sets] 
    #df['amd_scale'] = [PDD_to_AMD(pdd_1) if pdd_1 is not None else None for pdd_1 in pdds]
    df.to_pickle(f'{dfile}_AMD.pickle')

#    path_2 = 'CIFs_T2-crystals/Mg2Sn1S2Cl4_vdW_POSCAR.cif'
#    crystal_2 = pdd.read_cif(path_2)
#    pdd_2 = pdd.pdd(crystal_2, k)
#    print(crystal_2.name, ':', crystal_2.motif.shape[0], 'points in a unit cell with parameters', crystal_2.cellpar)
#    print(f'PDD (k={k}):')
#    print(pdd_2, '\n')
    
#    emd = pdd.emd(pdd_1, pdd_2)
#    f = time.time()
#    print(f'EMD distance (k={k}) between {crystal_1.name} and {crystal_2.name}:', emd)
