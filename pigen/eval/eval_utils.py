import os
import itertools
import numpy as np
import torch
import hydra

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra import initialize_config_dir, compose
from pathlib import Path
from pymatgen.core import Structure, Lattice, Composition, Element
from pigen.common.constants import CompScalerMeans, CompScalerStds
from pigen.common.data_utils import StandardScaler, chemical_symbols
from pigen.common.utils import get_default_device
from torch_geometric.data import DataLoader

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

device = get_default_device() 

a_radii = torch.tensor([0, 0.25, 0, 1.45, 1.05, 0.85, 0.7, 0.65, 0.6, 0.5, 0, 1.8, 1.5, 1.25, 1.1, 1.0, 1.0, 1.0, 0.71, 2.2, 1.8, 1.6, 1.4, 1.35, 1.4, 1.4, 1.4, 1.35, 1.35, 1.35, 1.35, 1.3, 1.25, 1.15, 1.15, 1.15, 0, 2.35, 2.0, 1.8, 1.55, 1.45, 1.45, 1.35, 1.3, 1.35, 1.4, 1.6, 1.55, 1.55, 1.45, 1.45, 1.4, 1.4, 0, 2.6, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.8, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.3, 1.35, 1.35, 1.35, 1.5, 1.9, 1.8, 1.6, 1.9, 0, 0, 0, 2.15, 1.95, 1.8, 1.8, 1.75, 1.75, 1.75, 1.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float, device=device)

# J-validity test, Jude's tool for charges
def j_validity(crys):
    try:
        formula = crys.structure.formula.replace(' ', '')
    except:
        formula = crys.formula.replace(' ','')

    result = assign_charges(formula)
    if result == []:
        return False
    else:
        return True

def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles

def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, map_location='cpu')
    return data

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


def get_model_path(eval_model_name):
    import diffcsp
    model_path = (
        Path(diffcsp.__file__).parent / 'prop_models' / eval_model_name)
    return model_path


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg

def load_model(model_path, load_data=False, testing=True):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt = None
            for ck in ckpts:
                if 'last' in ck.parts[-1]:
                    ckpt = str(ck)
            if ckpt is None:
                ckpt_epochs = np.array(
                    [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts if 'last' not in ckpt.parts[-1]])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hparams = os.path.join(model_path, "hparams.yaml")
        model = model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=False)
        try:
            model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
            model.scaler = torch.load(model_path / 'prop_scaler.pt')
        except:
            pass

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False,                
                scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                train_loader = datamodule.train_dataloader(shuffle=False)
                val_loader = datamodule.val_dataloader()[0]
                test_loader = (train_loader, val_loader)
        else:
            test_loader = None

    return model, test_loader, cfg


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list

def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    
    if isinstance(crys[0], dict):
        struc_fps = [c['struct_fp'] for c in crys]
        comp_fps = [c['comp_fp'] for c in crys]
        gt_struc_fps = [c['struct_fp'] for c in gt_crys if c['struct_fp'] is not None]
        gt_comp_fps = [c['comp_fp'] for c in gt_crys if c['comp_fp'] is not None]
    else:
        struc_fps = [c.struct_fp for c in crys]
        comp_fps = [c.comp_fp for c in crys]
        gt_struc_fps = [c.struct_fp for c in gt_crys]
        gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    for i,c in enumerate(gt_comp_fps):
        if len(c) != 132:
            print(i)
            print(len(c))

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None
