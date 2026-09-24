
from assets.diffusion import CSPDiffusion
from pathlib import Path
from torch_geometric.data import Data, Batch, DataLoader
from scripts.csp import SampleDataset
from pymatgen.core import Structure
from eval.eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from settings import *
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.vis.structure_vtk import StructureVis
from scripts.compute_metrics_old import Crystal
from ase.visualize import view
import numpy as np
# import imageio
import matplotlib.pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
from pymatgen.core.lattice import Lattice
import re
from ase.io import write
import torch
# from pymatviz.structure_viz import struct_2d

import glob
from PIL import Image

plt.rcParams['figure.dpi'] = 30

def sort_key(filename):
    # Extract the number from the filename
    number = re.search(r'struct_(\d+)', filename)
    if number is not None:
        return int(number.group(1))
    else:
        return 0

def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/struct_*.png"), key=sort_key)]    
    frame_one = frames[0]

    # Calculate the duration of each frame
    total_duration = 10000  # Total duration in milliseconds
    frame_duration = total_duration // len(frames)  # Duration of each frame in milliseconds

    frame_one.save("my_awesome.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=frame_duration, loop=0)

def diffusion_gif(loader, model, step_lr, random_state=42):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        # if torch.cuda.is_available():
        #     batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr, random_state=random_state)
    #     frac_coords.append(outputs['frac_coords'].detach().cpu())
    #     num_atoms.append(outputs['num_atoms'].detach().cpu())
    #     atom_types.append(outputs['atom_types'].detach().cpu())
    #     lattices.append(outputs['lattices'].detach().cpu())

    # frac_coords = torch.cat(frac_coords, dim=0)
    # num_atoms = torch.cat(num_atoms, dim=0)
    # atom_types = torch.cat(atom_types, dim=0)
    # lattices = torch.cat(lattices, dim=0)
    # lengths, angles = lattices_to_params_shape(lattices)
    return traj
    # return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)

def main(model_path: str = 'ckpts/mp_20/gen',
        formula: str = 'SrTiO3',
        save_path:str='gifs/',
        num_evals: int = 1,
        step_lr: float = 1e-5,
        random_state: int = 42,
        batch_size: int = 1):

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model_path = Path(model_path)
    model      = CSPDiffusion.load_from_checkpoint(model_path / 'last.ckpt',
                                                   **data_params,
                                                   **model_hparams, 
                                                   **scheduler_hparams, 
                                                   strict=False)

    # if torch.cuda.is_available():
    #     model.to('cuda')

    test_set = SampleDataset(formula, num_evals, random_state=random_state)
    test_loader = DataLoader(test_set, batch_size = min(batch_size, num_evals))

    traj = diffusion_gif(test_loader, model, step_lr, random_state=random_state)
    plot_structs_from_traj(traj, save_path=save_path)
# def sample_gif(batch, step_lr):


def plot_structs_from_traj(traj, save_path):
    all_frac_coords = traj['all_frac_coords']
    all_lattices = traj['all_lattices']
    all_atom_types = traj['atom_types']

    images = []  # List to store images

    for i, (frac_coords, lattices, atom_types) in enumerate(zip(all_frac_coords, all_lattices, all_atom_types)):
        if i %10 ==0 or i==0:
            lengths, angles = lattices_to_params_shape(lattices)
            lengths = lengths.squeeze().cpu().numpy()
            angles = angles.squeeze().cpu().numpy()
            atom_types = atom_types.cpu().numpy()
            frac_coords = frac_coords.cpu().numpy()

            structure = Structure(
                lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
                species=atom_types, coords=frac_coords, coords_are_cartesian=False)

            ase_atoms = AseAtomsAdaptor.get_atoms(structure)
            ase_atoms = ase_atoms * (2,2,2)

            fig, ax = plt.subplots()

            # Set the background color to white
            ax.set_facecolor('white')

            # Hide the axes
            ax.axis('off')

            # Plot the atoms
            plot_atoms(ase_atoms, ax, radii=0.3, rotation=('45x,45y,45z'))
            # Save the figure
            fig.savefig(f'{save_path}/struct_{i}.png')
            plt.close(fig)
        # # Save the structure as a PNG image
        # img_path = f'{save_path}/struct_{i}.png'
        # write(img_path, ase_atoms, rotation='45x,45y,45z', show_unit_cell=2)  # show_unit_cell=2 to display full unit cell

    # crys_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms=np.array([frac_coords.shape[0]]))
    # crystal   = Crystal(crys_list[0])

    print('Composition:', structure.composition)
    # print('Valid:', crystal.is_valid())
    
        # images.append(imageio.imread(img_path))  # Read the image and append to the list

    # Create a GIF
    # imageio.mimsave(f'{save_path}/structures.gif', images, duration=0.5)  # Adjust duration as needed

# def plot_structs_from_traj(traj, save_path):
#     all_frac_coords = traj['all_frac_coords']
#     all_lattices = traj['all_lattices']
#     all_atom_types = traj['atom_types']

#     for i, (frac_coords, lattices, atom_types) in enumerate(zip(all_frac_coords, all_lattices, all_atom_types)):
#         lengths, angles = lattices_to_params_shape(lattices)
#         lengths = lengths.squeeze().cpu().numpy()
#         angles = angles.squeeze().cpu().numpy()
#         atom_types = atom_types.cpu().numpy()
#         frac_coords = frac_coords.cpu().numpy()

#         structure = Structure(
#             lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
#             species=atom_types, coords=frac_coords, coords_are_cartesian=False)

#         ase_atoms = AseAtomsAdaptor.get_atoms(structure)
#         # Save the structure as  a PNG image
#         write(f'{save_path}/struct_{i}.png', ase_atoms, rotation='45x,45y,0z', show_unit_cell=2)  # show_unit_cell=2 to display full unit cell
#         # fig, ax = plt.subplots(figsize=(8,6))
#         # plot_structure(structure)
if __name__ == '__main__':
    main()
    # make_gif('gifs')