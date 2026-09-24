""" 
Sample crystal structures' (A, X, L) with some elements A fixed
E.g. Li-Li could be predefined in a composition, and remaining elements sampled
during diffusion inference 
"""

import os
import numpy as np
import time
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse
import yaml
from pymatgen.io.cif import CifWriter
import pandas as pd
from pathlib import Path
from pigen.assets.diffusion_pi import CSPDiffusion
from pigen.eval.eval_utils import get_crystals_list, lattices_to_params_shape, get_pymatgen
from pigen.settings import *

MAX_ATOMIC_NUM=100

train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301],
    
    'full_data': [0.0,
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
}

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

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
            
        # outputs, traj = model.sample(batch, step_lr = step_lr)
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
        self.total_num = 1000
        total_num = 1000
        #self.total_num    = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms    = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.env          = [chemical_symbols.index(f) for f in env]
        self.env_len      = len(self.env)
        self.is_carbon    = dataset == 'carbon_24'

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
    
def main(model_path='ckpts/compactness',
         save_path=None,
         num_batches_to_samples=10,
         batch_size=128,
         env=['Li','Li'],
         targets=[15.],
         guidance=None,
         dataset='full_data',
         step_lr=5e-6):
    
    # load_data if do reconstruction.
    model_path = Path(model_path)

    # Load the settings.yaml file
    with open(model_path / 'settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    ckpts = list(model_path.glob('*.ckpt'))
    print('.glob:', ckpts)
    assert len(ckpts) > 0, 'No checkpoint found in the model_path'

    ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
    print(ckpt_epochs)
    ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    print(ckpt)

    model = CSPDiffusion.load_from_checkpoint(ckpt, **settings, strict=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    test_set = SampleDataset(env, 
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

    elapsed_time = time.time() - start_time

    result = {'model_path': model_path,
            'time': elapsed_time,
            'num_batches_to_sample': num_batches_to_samples,
            'batch_size': batch_size,
            'env': env,
            'dataset': dataset,
            'step_lr': step_lr,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles}
    
    env_name = ''.join(env)

    if save_path is None:
        save_path = model_path

    crystal_list   = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    struct_list    = []
    for crys in crystal_list:
        crys['atom_types'] = np.argmax(crys['atom_types'], axis=-1) + 1
        structure = get_pymatgen(crys)
        struct_list.append(structure)
    
    strcuture_list = []
    for i,structure in enumerate(struct_list):
        formula = structure.composition.formula.replace(' ', '')
        if structure is not None:
            writer = CifWriter(structure)
            strcuture_list.append(writer.__str__())
        else:
            print(f"{i+1} Error Structure.")
   
    previs = '_'.join(env) 

    # Creating DataFrame
    df = pd.DataFrame({'structure': strcuture_list})
    df.to_csv(save_path/f'{env_name}_{targets}_{guidance}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating Inpainted structures using a trained model')
    parser.add_argument('--model_path', type=str, default='ckpts/dummy_ckpt',  help='Ckpt of trained model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to store generated strcutures.')
    parser.add_argument('--env', type=str, nargs='+', default=['Li','Li'], help='Local environment to be inpainted.')
    parser.add_argument('--dataset', type=str, default='mp_20', help='Refernce dataaset to define p(N) where N is the number of atoms in the material to be generated.')
    parser.add_argument('--step_lr', type=float, default=5e-6, help='Step LR for SMLD')
    parser.add_argument('--guidance', type=float, default=2.0, help='Guidance strength')
    parser.add_argument('--num_batches_to_samples', type=str, default=10, help='Num. of batches to sample.')
    parser.add_argument('--targets', type=float, nargs='+', default=[5.,2.], help='Targets for conditional generation. Need to follow the original order for training E.g., Iconf_cmpt [200.0, 0.7]')    
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    args = parser.parse_args()
    main(**vars(args))
