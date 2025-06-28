import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import omegaconf
from torch_geometric.data import Data
import torch.distributed as dist
import pickle
import numpy as np
from pigen.common.data_utils import preprocess_simple, preprocess_tensors
from pigen.common.data_utils import add_scaled_lattice_prop, get_scaler_from_data_list 
from pigen.common.data_utils import get_scalers_from_data_list
from pigen.settings import config

class SimpleCrystDataset(Dataset):
    '''
    Simplified version of the original CrysDataset in DiffCSP that avoids graph construction.
    '''
    def __init__(self, 
                 df: pd.DataFrame,
                 prop= 'formation_energy_per_atom',
                 cat_prop= False,
                 niggli= True, 
                 primitive= False,
                 preprocess_workers=30,
                 lattice_scale_method= 'scale_length', 
                 save_path= f'{config.PATHS.DATA_DIR}/dummy/train_ori.pt', 
                 tolerance= 0.1, 
                 use_space_group= False, 
                 use_pos_index= False,
                 target_energy=False,
                 gpus=1,
                 **kwargs):
        
        super().__init__()
        self.df = df
        self.prop = prop
        self.target_energy = target_energy

        if self.prop != ['']:
            self.df = self.df.dropna(subset=prop).reset_index(drop=True)

        self.niggli = niggli
        self.primitive = primitive
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.gpu = gpus

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

        self.lattice_scaler= get_scaler_from_data_list(self.cached_data, key='scaled_lattice')
        self.scaler        = get_scalers_from_data_list(self.cached_data, key=prop)
        self.cat_prop      = cat_prop

    def preprocess(self, save_path, preprocess_workers, prop):
       if os.path.exists(save_path):
           self.cached_data = torch.load(save_path, weights_only=False)
       else:
            cached_data = preprocess_simple(
            self.df,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            prop_list=prop, # list of properties to be scaled
            use_space_group=self.use_space_group,
            tol=self.tolerance,
            target_energy=self.target_energy)
            if self.gpu==1:
                torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        if self.prop == ['']:
            scaled_props = None
        else:
            if not self.cat_prop:
                scaled_props = np.column_stack([scaler.transform(data_dict[prop]) for prop, scaler in zip(self.prop, self.scaler)])
            else:
                scaled_props = np.array(data_dict[self.prop[0]]).reshape(-1,1)
                print('Running SimpleDataset scaled Props:', self.prop, self.scaler)
                print('scaled_props')
            
            scaled_props = torch.tensor(scaled_props, dtype=torch.float32)

        (frac_coords, atom_types, lengths, angles, num_atoms) = data_dict['crys_arrays']
        if self.target_energy:
            target_energy = np.array(data_dict['target_energy']).reshape(-1,1)
            
            data = Data(
                target_energy=torch.tensor(target_energy),
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                num_atoms=num_atoms,
                num_nodes=num_atoms,  
                y=scaled_props
                )

        else:
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                num_atoms=num_atoms,
                num_nodes=num_atoms, 
                y=scaled_props
                )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"
