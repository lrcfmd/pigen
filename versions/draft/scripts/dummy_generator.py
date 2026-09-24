"""
Generate random structures
Sampling lattice parameters from training data and sampling fractional coordinates from uniform distribution
and atom types from multinomial
"""

from common.data_utils import chemical_symbols, lattice_params_to_matrix
import numpy as np
from pymatgen.core import Structure
import pandas as pd
import torch

n_atoms_dist = [0.0,
                0.011397402156379998,
                0.03089184141268359,
                0.025288649291111298,
                0.07609941421173275,
                0.06039349690126496,
                0.0635134561507768,
                0.017987520163001952,
                0.055777230664742335,
                0.028822480685966552,
                0.06594362849138297,
                0.013901859241022158,
                0.07555819679089906,
                0.02162747262076577,
                0.052073605569233385,
                0.01173698955768741,
                0.050948722302402584,
                0.009699465149842941,
                0.03755624416334154,
                0.013095339162917056,
                0.07227905594702437,
                0.006887256982765939,
                0.03489260548433653,
                0.006420324305968249,
                0.06012819424399355,
                0.0024832328720604466,
                0.01825282282027337,
                0.004106885134561508,
                0.04562144494439256,
                0.007375413872145344,
                0.019239748705323034]

def get_lattice_dist(train_path='data/full_data/train.csv'):
    train = pd.read_csv(train_path)

    # train = train.iloc[:10] # for testing purposes
    lattice_par_train = []
    for cif in train['cif']:
        struct = Structure.from_str(cif, fmt='cif')
        lengths = struct.lattice.abc
        angles = struct.lattice.angles
        lattice_par_train.append(np.concatenate([lengths, angles]))
    
    # Stacking vertically
    lattice_par_train = np.vstack(lattice_par_train)

    # Calculate the mean and standard deviation of each lattice parameter
    means = np.mean(lattice_par_train, axis=0)
    std_devs = np.std(lattice_par_train, axis=0)

    return means, std_devs




def sample_structures(num_structures, 
                      mean_lattice, 
                      std_dev_lattice,
                      save_path='forcing_benchmark/', 
                      env=['Li','O']):
    frac_coords = []
    atom_types  = []
    lengths     = []
    angles      = []
    num_atoms   = []

    allowed_symbols = chemical_symbols[1:97] # remove 'X'
    one_hot_encoding = np.eye(len(allowed_symbols))
    env_atom_types = one_hot_encoding[[allowed_symbols.index(e) for e in env]]

    for i in range(num_structures):
        # Sample a number of atoms according to the distribution
        n_atoms_choices = np.arange(len(n_atoms_dist))
        n_atoms = np.random.choice(n_atoms_choices, p=n_atoms_dist)

        if n_atoms < len(env):
            n_atoms = len(env)

        # Sample the rest of atom types as one_hot vectors
        other_atom_nums = np.random.choice(len(allowed_symbols), size=n_atoms-len(env))
        other_atom_types = one_hot_encoding[other_atom_nums]

        at_types = np.concatenate([env_atom_types, other_atom_types], axis=0)

        # Sample new lattice parameters from these distributions
        new_lattice_params = np.random.normal(loc=mean_lattice, scale=std_dev_lattice)

        # Sample nfractional coordinates from a uniform distribution
        fractional_coords = np.random.rand(n_atoms, 3)

        # Converting to torch tensors
        atom_types.append(torch.tensor(at_types, dtype=torch.float32))
        lengths.append(torch.tensor(new_lattice_params[:3], dtype=torch.float32))
        angles.append(torch.tensor(new_lattice_params[3:], dtype=torch.float32))
        frac_coords.append(torch.tensor(fractional_coords, dtype=torch.float32))
        num_atoms.append(torch.tensor(n_atoms, dtype=torch.int32))

    # Stacking vertically
    atom_types = torch.vstack(atom_types)
    lengths = torch.vstack(lengths)
    angles = torch.vstack(angles)
    frac_coords = torch.vstack(frac_coords)
    num_atoms = torch.hstack(num_atoms)

    result = {'frac_coords': frac_coords, 'num_atoms': num_atoms, 'atom_types': atom_types, 'lengths': lengths, 'angles': angles}

    env_name = ''.join(env)
    torch.save(result, save_path + f'eval_dummy_{env_name}.pt')


if __name__ == '__main__':
    mean_lattice, std_dev_lattice = get_lattice_dist()
    sample_structures(num_structures=1280,
                      mean_lattice=mean_lattice, 
                      std_dev_lattice=std_dev_lattice, 
                      env=['Li','Li'])