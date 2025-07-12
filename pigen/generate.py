import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifWriter
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union 
import yaml

from pigen.assets.diffusion_pi import CSPDiffusion
from pigen.common.constants import TRAIN_DIST
from pigen.common.utils import set_logger
from pigen.eval.eval_utils import (
        get_crystals_list,
        get_pymatgen,
        lattices_to_params_shape)
from pigen.settings import PROJECT_ROOT, config

RECOMMENDED_STEP_LR = {'gen': {
                              'full_data':5e-6}
                              }

def diffusion(
    logger: logging.Logger,
    loader: DataLoader,
    model: Union[torch.nn.Module],
    step_lr: float,
    guidance: Optional[float] = None,
    targets: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Runs the diffusion-based sampling process over a dataset loader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader providing input batches for sampling.
    model : torch.nn.Module
        The model used for (conditional) sampling.
    step_lr : float
        The learning rate used for the sampling steps.
    guidance : float or None, optional
        Strength of guidance signal for conditional sampling. Used only if `targets` is provided.
    targets : list or None, optional
        Target values for conditional sampling. If None or empty, unconditional sampling is used.

    Returns
    -------
    tuple
        Tuple containing:
        - frac_coords (Tensor): Fractional coordinates of atoms.
        - atom_types (Tensor): Atomic numbers/types.
        - lattices (Tensor): Lattice matrices.
        - lengths (Tensor): Lattice lengths.
        - angles (Tensor): Lattice angles.
        - num_atoms (Tensor): Number of atoms in each structure.
    """

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch = batch.cuda() 
        
        if targets is None or targets == '':
            outputs, traj = model.sample(batch, step_lr=step_lr)
        else:
            logger.info(f'Using conditional sampling, guidance: {guidance}, targets: {targets}')

            if torch.cuda.device_count() > 1:
                outputs, traj = model.module.conditional_sample(batch, step_lr=step_lr, guidance=guidance, targets=targets)
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

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)

class SampleDataset(Dataset):
    """
    Dataset sampling atomic structures with number of atoms drawn from a predefined distribution.
    """

    def __init__(self, dataset: str, total_num: int):
        super().__init__()
        self.total_num: int = total_num
        self.distribution: List[float] = TRAIN_DIST[dataset]
        self.num_atoms: np.ndarray = np.random.choice(
            len(self.distribution),
            total_num,
            p=self.distribution
        )

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index: int) -> Data:
        num_atom: int = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        return data

def main(
    model_path,
    save_path=None,
    dataset='full_data',
    step_lr=5e-6,
    num_batches_to_samples=1,
    guidance=2,
    targets=[0.],
    batch_size=3
):
    """
    Run diffusion sampling using a pretrained CSPDiffusion model checkpoint.

    Args:
        model_path (str or Path): Directory containing the model checkpoint and settings.yaml.
        save_path (str or Path, optional): Directory to save the results CSV. Defaults to model_path.
        dataset (str): Dataset name to use for SampleDataset distribution.
        step_lr (float): Step learning rate for diffusion sampling. If <0, uses recommended value.
        num_batches_to_samples (int): Number of batches to sample from.
        guidance (float or int): Guidance scale for conditional sampling.
        targets (list): Target property values for conditional sampling.
        batch_size (int): Batch size for DataLoader.

    Raises:
        ValueError: If no checkpoint files are found in model_path.

    Returns:
        None: Saves results to CSV and logs progress.
    """
    logger = set_logger(model_path, __name__, 'INFO')
    model_path = Path(model_path)

    logger.info(f'Loading model checkpoint from: {model_path}')
    with open(model_path / 'settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)

    ckpts = list(model_path.glob('*.ckpt'))
    if not ckpts:
        logger.error(f'No checkpoint found in {model_path}')
        raise ValueError(f'No checkpoint found in {model_path}')

    ckpt_epochs = np.array([
        int(ckpt.parts[-1].split('-')[0].split('=')[1])
        for ckpt in ckpts
    ])
    latest_ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    logger.info(f'Using latest checkpoint: {latest_ckpt}')

    model = CSPDiffusion.load_from_checkpoint(latest_ckpt, **settings, strict=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f'CUDA available (Torch {torch.__version__}), setting device to CUDA')
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    else:
        logger.info('CUDA not available, running on CPU')

    total_runs = batch_size * num_batches_to_samples
    logger.info(f'Sampling diffusion model for {total_runs} total runs')

    test_set = SampleDataset(dataset, total_num=total_runs)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    logger.info(f'Dataset and DataLoader created with batch_size={batch_size}')

    step_lr = step_lr if step_lr >= 0 else RECOMMENDED_STEP_LR['gen'][dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(
         logger, test_loader, model, step_lr, guidance, targets
         )
    elapsed_time = time.time() - start_time
    logger.info(f'Diffusion sampling completed in {elapsed_time:.2f} seconds')

    results = {
        'eval_setting': {
            'model_path': model_path,
            'time': time.time(),
            'num_batches_to_sample': num_batches_to_samples,
            'batch_size': batch_size,
            'dataset': dataset,
            'step_lr': step_lr
        },
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles
    }

    if save_path is None:
        save_path = model_path
    save_path = Path(save_path)
    logger.info(f'Saving results to: {save_path}')

    logger.info('Generating crystal list')
    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    struct_list = []
    for crys in crystal_list:
        crys['atom_types'] = np.argmax(crys['atom_types'], axis=-1) + 1
        structure = get_pymatgen(crys)
        struct_list.append(structure)

    structure_str_list = []
    for i, structure in enumerate(struct_list):
        formula = structure.composition.formula.replace(' ', '')
        if structure is not None:
            writer = CifWriter(structure)
            structure_str_list.append(writer.__str__())
        else:
            logger.info(f"{i + 1} Error: invalid structure.")

    df = pd.DataFrame({'structure': structure_str_list})
    output_file = save_path / f'denovo_{targets}_{guidance}.csv'
    df.to_csv(output_file, index=False)
    logger.info(f'Saved structures CSV to {output_file}')

def parse_args():
    parser = argparse.ArgumentParser(description='Generating De Novo structures using a trained model')
    parser.add_argument('--model_path', type=str, default=config.PATHS.CHECKPOINT_DIR,  help='Ckpt of trained model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to store generated structures.')
    parser.add_argument('--dataset', type=str, default='full_data', help='Refernce dataset to define p(N) where N is the number of atoms in the material to be generated.')
    parser.add_argument('--step_lr', type=float, default=5e-6, help='Step LR for SMLD')
    parser.add_argument('--guidance', type=float, default=2.0, help='Guidance strength')
    parser.add_argument('--num_batches_to_samples', type=str, default=10, help='Num. of batches to sample.')
    parser.add_argument('--targets', type=float, nargs='+', default=[9.0, 0.7], help='Targets for conditional generation. Need to follow the original order for training E.g., Entropy_cmpt [9.0, 0.7].')    
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    return  parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
