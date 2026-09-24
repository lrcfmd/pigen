import os
import sys
import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition, Element
import matplotlib.pyplot as plt

def writecif(cif):
    with open('tmp.cif', 'w') as f:
        f.write(cif)

def volume(element):
    r = Element(element).atomic_radius
    if r is None:
        return 0
    else:
        return 1/3 * 4 * np.pi * r ** 3

def atomicvolumes(s):
    if isinstance(s, str):
        s = Structure.from_str(s, fmt='cif')

    elements = Composition(s.formula).as_dict()
    volumes = np.sum([volume(e) * k for e, k in elements.items()])
    return round(volumes / s.volume, 2)


if __name__ == '__main__':
    fdir = sys.argv[1]
    compact = []
    compositions = []
    for p, d, fi in os.walk(f'{fdir}'):
        for cif in fi:
            if '.cif' in cif:
                structure = Structure.from_file(f'{fdir}/{cif}')
                #compositions.append(structure.composition.reduced_formula)
                compositions.append(cif[:-4])
                compact.append(atomicvolumes(structure))

    fdi = fdir.strip('/')
    df = pd.DataFrame({'composition':compositions, 'compactness':compact})
    #df.to_csv(f'{fdi}_av{round(np.average(compact),2)}.csv', index=False)
    #sys.exit(0)

    print('average', np.average(compact))
    plt.hist(compact, bins=20, label=f'{fdir}; \n average = {round(np.average(compact), 2)}')
    plt.legend()
    plt.show()
