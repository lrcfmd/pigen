import torch
from p_tqdm import p_map
import pickle
import os
from pymatgen.io.cif import CifWriter

# labels= ["LiLi_[0.7]_2.0",
#         "LiLi_[0.7]_3.0",
#         "LiO_[0.7]_2.0",
#         "LiO_[0.7]_3.0",
#         "LiS_[0.7]_2.0",
#         "LiS_[0.7]_3.0"]
labels = ['LiLi_[0.7]_0.0',
          'LiO_[0.7]_0.0',
          'LiS_[0.7]_0.0']

for label in labels:
    with open(f'ckpts/compactness/crys_pcsp_{label}.pkl', 'rb') as f:
        gen_crys = pickle.load(f)
    
    gen_crys_valid = [s for s in gen_crys if s['valid']]
    
    if not os.path.exists(f'ckpts/compactness/cifs_{label}'):
        os.makedirs(f'ckpts/compactness/cifs_{label}')
    
    for crys in gen_crys_valid:
        struct = crys['structure']
        formula = struct.composition.formula.replace(' ','')
        writer = CifWriter(struct)
        writer.write_file(f'ckpts/compactness/cifs_{label}/{formula}.cif')