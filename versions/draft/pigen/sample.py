from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import chemparse
from p_tqdm import p_map
import os
import time
import argparse
import torch
from pathlib import Path
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pigen.settings import *
from pigen.assets.csp_diffusion import CSPDiffusion
from pigen.pymatgen.io.cif import CifWriter
from pigen.eval.eval_utils import lattices_to_params_shape, get_crystals_list, get_pymatgen
from pigen.common.data_utils import chemical_symbols
from pigen.common.utils import set_logger

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
            outputs, traj = model.sample(batch, step_lr = step_lr)
        else:
            outputs, traj = model.conditional_sample(batch, step_lr=step_lr, guidance=guidance, targets=targets)

        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):

    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )

def main(args):
    model_path = Path(args.model_path)
    logger = set_logger(model_path, __name__, 'INFO')   
    logger.info(f'Loading model ckpt from model path: {model_path}')

    with open(model_path / 'settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)

    ckpts = list(model_path.glob('*.ckpt'))
    logger.info(f'.glob: {ckpts}')
    if len(ckpts) == 0:
        logger.error('No checkpoint found in the model_path')
        raise ValueError('No checkpoint found in the {model_path}')

    ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
    logger.info(f'Number of ckpt epochs: {ckpt_epochs}')
    ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])

    logger.info(f'Loading model from ckpt {ckpt}')
    model = CSPDiffusion.load_from_checkpoint(ckpt, **settings, strict=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        logger.info(f'CUDA is avalable. Torch v: {torch.__version__}. Cuda is set as device')

    logger.info(f'Evaluating the diffusion model for {args.num_evals} args.num_evals')

    test_set = SampleDataset(args.formula, args.num_evals)
    test_loader = DataLoader(test_set, batch_size = max(args.batch_size, args.num_evals))
    logger.info(f'SampleDataset and DataLoader are set with batch_size: {args.batch_size}')

    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, 
                                                                                model, 
                                                                                args.step_lr, 
                                                                                args.guidance, 
                                                                                args.targets)

    if args.save_path is None:
        save_path = model_path
    logger.info(f'Saving results to save_path: {save_path}')

    crystal_list   = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    struct_list    = []
    for crys in crystal_list:
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

    df = pd.DataFrame({'structure': strcuture_list})
    df.to_csv(save_path/f'csp_{args.formula}.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--formula', required=True)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--step_lr', default=5e-6, type=float)
    parser.add_argument('--targets', default=None, type=float)
    parser.add_argument('--guidance', default=2, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    parse_args()
    main(**vars(args))
