from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

DATA_DIR = Path('/Users/andrij/Programmes/Contrastive_learning_MPDS_structure_space/PCD_DATA/')
amds_all = []
chunk_files = sorted(DATA_DIR.glob('periodic_sets_chunk_*.pickle'))

for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
    with open(chunk_path, 'rb') as f:
        periodic_sets = pickle.load(f)
        print(chunk_path, len(periodic_sets['periodic_sets']), periodic_sets.keys())
        tags = [i.types  if i is not None else None for i in periodic_sets['periodic_sets']]
        print('Tags types:', len(tags), 'Not none:', len([i for i in tags if i is not None]))
