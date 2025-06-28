""" 
Sample crystal structures' (A, X, L) with some elements A fixed
E.g. Li-Li could be predefined in a composition, and remaining elements sampled
during diffusion inference 
"""

import argparse
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifWriter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml

from pigen.assets.diffusion_pi import CSPDiffusion
from pigen.common.constants import train_dist
from pigen.common.data_utils import chemical_symbols
from pigen.common.utils import set_logger
from pigen.eval.eval_utils import (
        get_crystals_list,
        get_pymatgen,
        lattices_to_params_shape)
from pigen.settings import PROJECT_ROOT, config

MAX_ATOMIC_NUM=100

def diffusion(loader, model, step_lr, guidance=None, targets=None):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        
        if targets is None:
            outputs,traj  = model.fill(batch, step_lr = step_lr)
        else:
            outputs, traj = model.conditional_fill(batch, step_lr = step_lr, guidance=guidance, targets=targets)
            
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)

class SampleDataset(Dataset):
    def __init__(self, 
                 env,
                 dataset,
                 total_num):
        
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms    = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.env          = [chemical_symbols.index(f) for f in env]
        self.env_len      = len(self.env)

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        env      = torch.LongTensor(self.env)
        env_len  = torch.LongTensor([self.env_len])
        data     = Data(num_atoms=torch.LongTensor([num_atom]),
                        env=env,
                        env_len=env_len,
                        num_nodes=num_atom)
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data
    
def main(model_path,
         save_path=None,
         num_batches_to_samples=10,
         batch_size=128,
         chemical_env=['Li','Li'],
         targets=[9.],
         guidance=None,
         dataset='full_data',
         step_lr=5e-6):


    # load_data if do reconstruction.
    model_path = Path(model_path)
    logger = set_logger(model_path, __name__, 'INFO')
    logger.info(f'Loading model ckpt from model path: {model_path}')


    # Load the settings.yaml file
    with open(model_path / 'settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    ckpts = list(model_path.glob('*.ckpt'))
    logger.info('.glob:', ckpts)
    if len(ckpts) == 0:
        logger.error('No checkpoint found in the model_path')
        raise ValueError('No checkpoint found in the {model_path}')

    ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
    ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    logger.info(f'Loading model from ckpt {ckpt}')
    model = CSPDiffusion.load_from_checkpoint(ckpt, **settings, strict=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    logger.info(f'Evaluating the diffusion model for {batch_size * num_batches_to_samples} samples')
    test_set = SampleDataset(chemica_env, 
                             dataset,
                             batch_size * num_batches_to_samples)
    
    test_loader = DataLoader(test_set, batch_size = batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, 
     lengths, angles, num_atoms) = diffusion(test_loader, 
                                             model, 
                                             step_lr,
                                             guidance=guidance,
                                             targets =targets)

    env_name = ''.join(chemical_env)

    if save_path is None:
        save_path = model_path
    logger.info(f'Saving results to save_path: {save_path}'

    crystal_list   = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    struct_list    = []
    for crys in crystal_list:
        crys['atom_types'] = np.argmax(crys['atom_types'], axis=-1) + 1
        structure = get_pymatgen(crys)
        struct_list.append(structure)
    
    structure_list = []
    for i,structure in enumerate(struct_list):
        formula = structure.composition.formula.replace(' ', '')
        if structure is not None:
            writer = CifWriter(structure)
            structure_list.append(writer.__str__())
        else:
            logger.info(f"{i+1} Error Structure.")
   
    # Creating DataFrame
    df = pd.DataFrame({'structure': structure_list})
    df.to_csv(save_path/f'{env_name}_{targets}_{guidance}.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Generating Inpainted structures using a trained model')
    parser.add_argument('--model_path', type=str, default=f'{config.LOG_DIR}/dummy_ckpt',  help='Ckpt of trained model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to store generated structures.')
    parser.add_argument('--env', type=str, nargs='+', default=['Li','Li'], help='Local environment to be inpainted.')
    parser.add_argument('--dataset', type=str, default='full_data', help='Refernce dataset to define p(N) where N is the number of atoms in the material to be generated.')
    parser.add_argument('--step_lr', type=float, default=5e-6, help='Step LR for SMLD')
    parser.add_argument('--guidance', type=float, default=2.0, help='Guidance strength')
    parser.add_argument('--num_batches_to_samples', type=int, default=10, help='Num. of batches to sample.')
    parser.add_argument('--targets', type=float, nargs='+', default=[9, 0.7], help='Targets for conditional generation. Need to follow the original order for training E.g., Entropy_cmpt [9, 0.7]')    
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
