"""
This module is adapted from https://github.com/jiaor17/DiffCSP,
originally implementing a Denoising Diffusion Probabilistic Model
for crystal structure prediction.

Modifications include physics-informed logic for equivariant
diffusion training, and integration with classifier-free guidance
for targeted-property (including chemical and structural diversity)
generation of crystal structures.

Original License: MIT
"""

import math
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from pigen.assets.cspnet import CSPNet
from pigen.common.diff_utils import BetaScheduler, SigmaScheduler
from pigen.common.data_utils import lattice_params_to_matrix_torch
from pigen.common.diff_utils import d_log_p_wrapped_normal
from pigen.eval.eval_utils import get_crystals_list, a_radii
from pigen.eval.eval_utils import lattices_to_params_shape

MAX_ATOMIC_NUM=100

import torch
from typing import List, Optional

def make_type_mask(allowed_atomic_numbers: List[int], max_atomic_num: int, device) -> torch.Tensor:
    """
    Boolean mask of shape [max_atomic_num], True = allowed.
    Atom types are stored 0-indexed in the logit vector (argmax + 1 = atomic number),
    so allowed_atomic_numbers=[3, 8] -> indices [2, 7].
    """
    mask = torch.zeros(max_atomic_num, dtype=torch.bool, device=device)
    idx = torch.tensor([z - 1 for z in allowed_atomic_numbers], device=device)
    mask[idx] = True
    return mask


def apply_type_mask(t: torch.Tensor, allowed_mask: torch.Tensor, neg_value: float = -1e4) -> torch.Tensor:
    """t: [N_atoms, MAX_ATOMIC_NUM] logits. Sets disallowed channels to a large negative constant."""
    t = t.clone()
    t[:, ~allowed_mask] = neg_value
    return t

def weighted_loss_component(
    loss_cmpt: torch.Tensor,
    base_threshold: float = 1.0,
    extreme_threshold: float = 10.0
) -> torch.Tensor:
    """
    Applies weighted truncation to loss values based on their magnitude.

    Losses below `base_threshold` are left unchanged.
    Losses between `base_threshold` and `extreme_threshold` are log-scaled.
    Losses above `extreme_threshold` are log10-scaled to suppress extreme gradients.

    Args:
        loss_cmpt (torch.Tensor): 1D tensor of loss components.
        base_threshold (float): Threshold below which values are considered normal.
        extreme_threshold (float): Threshold above which values are treated as extreme.

    Returns:
        torch.Tensor: Scalar mean of the adjusted loss values.

    Notes:
        Experimental settings tested:
            - (1, 1, 1): optimal and default
            - (3, 1, 1): smoother training, worse eval
            - (3, 0.5, 0.5): smoother training, worse eval
            - (3, 0.3, 0.3): smoother training, worse eval
    """
    normal_mask = loss_cmpt <= base_threshold
    high_mask = (loss_cmpt > base_threshold) & (loss_cmpt <= extreme_threshold)
    extreme_mask = loss_cmpt > extreme_threshold

    normal_vals = loss_cmpt[normal_mask]
    high_vals = torch.log1p(loss_cmpt[high_mask])
    extreme_vals = torch.log10(loss_cmpt[extreme_mask] + 1)

    total_loss = torch.cat([normal_vals, high_vals, extreme_vals])
    return total_loss.mean()

def compute_volume(batch_lattice: torch.Tensor) -> torch.Tensor:
    """
    Compute unit cell volumes from a batch of lattice matrices.

    Args:
        batch_lattice (torch.Tensor): Tensor of shape (N, 3, 3),
            where each (3, 3) matrix represents lattice vectors as rows or columns.

    Returns:
        torch.Tensor: Tensor of shape (N,) containing scalar volumes of each lattice.
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum('bi,bi->b', vector_a,
                                  torch.cross(vector_b, vector_c, dim=1)))


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = optim.Adam(params=self.parameters())
        if not self.hparams.optim['use_lr_scheduler']:
            return [opt]
        scheduler = ReduceLROnPlateau(optimizer=opt,
                                      factor=self.hparams.optim['lr_factor'],
                                      patience=self.hparams.optim['lr_patience'],
                                      min_lr=self.hparams.optim['min_lr'])
        
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"} 

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GradientLogger(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        # Log gradients of cond_emb
        for name, param in pl_module.decoder.cond_emb.named_parameters():
            if param.grad is not None:
                trainer.logger.experiment.log({f'{name}_param': param.mean(),
                                               f'{name}_grad' : param.grad.mean(),
                                               f'{name}_param_norm': param.norm(),
                                             f'{name}_grad_norm': param.grad.norm()})


class CSPDiffusion(BaseModule):
    """
    A diffusion model for crystal structure generation, integrating physics-informed 
    constraints and property prediction via CSPNet.

    Attributes:
        decoder (CSPNet): Neural network for decoding latent states into structure and properties.
        beta_scheduler (BetaScheduler): Controls noise schedule for forward process.
        sigma_scheduler (SigmaScheduler): Controls noise level for reverse sampling.
        time_embedding (SinusoidalTimeEmbeddings): Time-step embedding module.
        keep_lattice (bool): Whether to freeze lattice parameters based on cost weight.
        keep_coords (bool): Whether to freeze fractional coordinates based on cost weight.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fwd_step = 0
        self.decoder = CSPNet(
            latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
            ln=True,
            pred_type=True,
            p_cond=self.hparams.p_cond,
            prop_weights=self.hparams.prop_weights,
            smooth=True,
            n_props=len(self.hparams.prop)
        )

        self.beta_scheduler  = BetaScheduler(timesteps=self.hparams.timesteps,
                                             scheduler_mode=self.hparams.scheduler_mode)

        self.sigma_scheduler = SigmaScheduler(timesteps=self.hparams.timesteps)
        self.time_dim        = self.hparams.time_dim
        self.time_embedding  = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice    = self.hparams.cost_lattice < 1e-5
        self.keep_coords     = self.hparams.cost_coord < 1e-5
        self._log2           = torch.log(torch.tensor(2.0))

    def log_cosh_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) + F.softplus(-2 * torch.abs(x)) - self._log2

    def calculate_cmpt(self, type_prob, pred_l, batch_idx):
        radii_table  = a_radii.to(device=type_prob.device, dtype=torch.float32)[:type_prob.shape[1]]
        expected_r3  = (type_prob * radii_table[None, :] ** 3).sum(dim=-1)
        atomic_vols  = (4.0 / 3.0) * torch.pi * expected_r3
        cell_volumes = compute_volume(pred_l).clamp(min=1.0).to(torch.float32)  # 1 Å³ floor
        volumes_a    = torch.zeros(cell_volumes.shape[0], device=type_prob.device, dtype=torch.float32)
        volumes_a.scatter_add_(0, batch_idx, atomic_vols)
        packing_frac = volumes_a / cell_volumes
        return torch.nan_to_num(packing_frac, nan=1.0, posinf=1e4, neginf=0.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the CSPDiffusion model.

        Args:
            batch (Dict[str, Tensor]): A dictionary containing input tensors,
                typically including coordinates, lattice, atom types, and conditioning info.

        Returns:
            Dict[str, Tensor]: Dictionary containing individual loss components:
                - 'loss': total loss
                - 'loss_lattice': loss related to lattice prediction
                - 'loss_coord': loss related to fractional coordinates
                - 'loss_type': atom type prediction loss
                - 'loss_cmpt': compactness regularization loss
        """ 
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        diff_step = (times * self.beta_scheduler.timesteps).long()

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod) #mean of t noisy dist
        c1 = torch.sqrt(1. - alphas_cumprod) #std of t noisy dist

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        rand_t = torch.randn_like(gt_atom_types_onehot)

        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, pred_t = self.decoder(time_emb,
                                              atom_type_probs, 
                                              input_frac_coords, 
                                              input_lattice, 
                                              batch.num_atoms, 
                                              batch.batch,
                                              condition=batch.y,
                                              stage='train')

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_type = F.mse_loss(pred_t, rand_t)

        # ── compactness loss ─────────────────────────────────────────────────────
        # Lattice: fully detached — no gradient conflict with loss_lattice
        with torch.no_grad():
            c0_safe          = c0.clamp(min=0.1)
            c1_safe          = c1
            denoised_lattice = (input_lattice - c1_safe[:, None, None] * pred_l) \
                               / c0_safe[:, None, None]

        # Types: gradient flows through pred_t, gated by signal-to-noise
        c0_atoms       = c0.repeat_interleave(batch.num_atoms)[:, None]
        c1_atoms       = c1.repeat_interleave(batch.num_atoms)[:, None]
        denoised_types = (atom_type_probs - c1_atoms * pred_t) / c0_atoms.clamp(min=0.1)
        type_prob      = denoised_types.softmax(dim=-1)

        pred_cmpt   = self.calculate_cmpt(type_prob, denoised_lattice, batch.batch)
        target_cmpt = batch.target_energy.squeeze(-1).float()
        pred_cmpt   = torch.nan_to_num(pred_cmpt,   nan=0.0, posinf=1e4, neginf=0.0)
        target_cmpt = torch.nan_to_num(target_cmpt, nan=0.0, posinf=1e4, neginf=0.0)

        # gate: alphas_cumprod ~ 0 at t=T (noisy, ignore), ~ 1 at t=0 (clean, trust)
        cmpt_gate     = alphas_cumprod.detach()
        loss_cmpt_raw = self.log_cosh_loss(pred_cmpt - target_cmpt)   # [batch_size]
        loss_cmpt     = (cmpt_gate * loss_cmpt_raw).mean()

        # ── total loss ───────────────────────────────────────────────────────────
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord   * loss_coord   +
            self.hparams.cost_type    * loss_type    +
            self.hparams.cost_cmpt    * loss_cmpt
        )
        self._fwd_step += 1
        if self._fwd_step % 100 == 0:  # print every 100 steps
            with torch.no_grad():
#               self.logger.experiment.add_scalars('cmpt_debug', {
#                   'c0_min':          c0.min().item(),
#                   'c0_max':          c0.max().item(),
#                   'pred_cmpt_min':   pred_cmpt.min().item(),
#                   'pred_cmpt_max':   pred_cmpt.max().item(),
#                   'loss_cmpt':       loss_cmpt.item(),
#                   'loss_lattice':    loss_lattice.item(),
#                   'cmpt_gate_max':   (cmpt_gate * loss_cmpt_raw).max().item(),
#               }, global_step=self._fwd_step)
               print(
                   f"[step {self._fwd_step}] "
                   f"c0=[{c0.min():.3f},{c0.max():.3f}] "
                   f"pred_cmpt=[{pred_cmpt.min():.3f},{pred_cmpt.max():.3f}] "
                   f"loss_cmpt={loss_cmpt.item():.4f} "
                   f"loss_lattice={loss_lattice.item():.4f} "
                   f"gate_max={((cmpt_gate * loss_cmpt_raw).max().item()):.4f}"
               )

        return {'loss'         : loss,
                'loss_lattice' : loss_lattice,
                'loss_coord'   : loss_coord,
                'loss_type'    : loss_type,
                'loss_cmpt'    : loss_cmpt
            }

    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['lattice_scaler'] = self.lattice_scaler
        checkpoint['scaler']         = self.scaler
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            self.lattice_scaler = checkpoint['lattice_scaler']
            self.scaler         = checkpoint['scaler']
        except:
            pass
    
    @torch.no_grad()
    def diffuse_atom_types(self, atom_types, t):
        alphas         = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        c0             = torch.sqrt(alphas_cumprod)
        c1             = torch.sqrt(1. - alphas_cumprod) 
        rand_t         = torch.randn_like(atom_types.float()) #atom types are LongTensor
        atom_type_probs= (c0.repeat_interleave(atom_types.shape[0])[:, None] * atom_types + c1.repeat_interleave(atom_types.shape[0])[:, None] * rand_t)
        
        return atom_type_probs

    @torch.no_grad()
    def fill(
        self,
        batch,
        diff_ratio: float = 1.0,
        step_lr: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs reverse diffusion to generate atom types, fractional coordinates, and lattices
        conditioned on the batch.

        Args:
            batch: Batched input data containing atomic environments, number of atoms,
                   and other properties.
            diff_ratio (float): Ratio controlling noise scale during generation.
            step_lr (float): Step size for the reverse diffusion schedule.

        Returns:
            traj_0 (Dict[str, Tensor]): Final output at step t=0 containing:
                - 'num_atoms': Tensor of shape (B,)
                - 'atom_types': Tensor of shape (N, MAX_ATOMIC_NUM)
                - 'frac_coords': Tensor of shape (N, 3)
                - 'lattices': Tensor of shape (B, 3, 3)

            traj_stack (Dict[str, Tensor]): Trajectory over all reverse steps:
                - 'num_atoms': Same as above
                - 'atom_types': (T+1, N) with predicted atomic numbers
                - 'all_frac_coords': (T+1, N, 3)
                - 'all_lattices': (T+1, B, 3, 3)
        """

        batch_size       = batch.num_graphs
        l_T, x_T         = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T              = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        envs             = batch.env.view(batch_size, -1)
        n_envs           = envs.shape[1]

        cumulative_sum   = torch.cumsum(batch.num_atoms, dim=0)
        starting_indices = torch.cat([torch.tensor([0]).to(self.device), cumulative_sum[:-1]])

        N                = torch.min(batch.num_atoms, torch.tensor([envs.shape[1]]).to(self.device).repeat(len(batch.num_atoms))).cpu().numpy()

        # Initialize the tensor to hold the repeated env_block values
        env_atom_types = torch.zeros_like(t_T)

        # initialize fwd_mask as zeros
        fwd_mask = torch.zeros_like(t_T)  

        for idx, length in enumerate(N):
            start_idx = starting_indices[idx]
            end_idx = start_idx + length

            env_atom_types[start_idx:end_idx, :] = F.one_hot(envs[idx, :length] - 1, MAX_ATOMIC_NUM)
            fwd_mask[start_idx:end_idx, :] = 1

        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T, #initial data just noise at T=1000
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times    = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas         = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            #these are for reverse process..
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords'] #noisy version of frac coords at time step t
            l_t = traj[t]['lattices'] #noisy version of lattices at time step t
            t_t = traj[t]['atom_types'] #noisy version of types at time step t

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # Corrector (this only accounts for fractional coordinates (?))
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
        
            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))  

            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            # Denoised atom types
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            # Denoised atom types
            fwd_atom_types = self.diffuse_atom_types(env_atom_types, t-1)
            t_t_minus_1 = fwd_mask * fwd_atom_types + (1 - fwd_mask) * t_t_minus_1

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }
        
        return traj[0], traj_stack
    
    @torch.no_grad()
    def conditional_fill(
        self,
        batch,
        diff_ratio: float = 1.0,
        step_lr: float = 1e-5,
        guidance: float = 3.0,
        targets: List[float] = [4.0, 2.0]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Reverse diffusion with classifier-free guidance conditioned on target properties.
        
        Args:
            batch: Input batch with atomic info, lattice, etc.
            diff_ratio (float): Noise ratio during sampling.
            step_lr (float): Step size for predictor-corrector schedule.
            guidance (float): Guidance strength; higher means more reliance on conditioning.
            targets (List[float]): List of target values for each conditioned property,
                                   scaled internally using self.scaler.
        
        Returns:
            traj_0 (Dict): Final structures at t=0.
            traj_stack (Dict): All structures over trajectory (T+1 steps).
        """

        batch_size = batch.num_graphs

        conditions = [] 
        for c,scaler in zip(targets,self.scaler):
            c = torch.tensor([[c]]*len(batch), dtype=torch.float32).view(-1,1).to(self.device)
            c = scaler.transform(c)
            conditions.append(c)
        
        conditions = torch.hstack(conditions)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,                  #initial data just noise at T=1000
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        envs             = batch.env.view(batch_size, -1)
        n_envs           = envs.shape[1]

        cumulative_sum   = torch.cumsum(batch.num_atoms, dim=0)
        starting_indices = torch.cat([torch.tensor([0]).to(self.device), cumulative_sum[:-1]])

        N                = torch.min(batch.num_atoms, torch.tensor([envs.shape[1]]).to(self.device).repeat(len(batch.num_atoms))).cpu().numpy()

        # Initialize the tensor to hold the repeated env_block values
        env_atom_types = torch.zeros_like(t_T)

        # initialize fwd_mask as zeros
        fwd_mask = torch.zeros_like(t_T)  

        for idx, length in enumerate(N):
            start_idx = starting_indices[idx]
            end_idx = start_idx + length

            env_atom_types[start_idx:end_idx, :] = F.one_hot(envs[idx, :length] - 1, MAX_ATOMIC_NUM)
            fwd_mask[start_idx:end_idx, :] = 1

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times    = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords'] #noisy version of frac coords at time step t
            l_t = traj[t]['lattices'] #noisy version of lattices at time step t
            t_t = traj[t]['atom_types'] #noisy version of types at time step t

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            
            _, pred_x_u, _ = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=None, stage='eval')
            _,pred_x_c, _  = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=conditions, stage='eval')
            pred_x = (1+guidance)*pred_x_c - guidance*pred_x_u

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l_u, pred_x_u, pred_t_u = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=None, stage='eval')
            pred_l_c, pred_x_c, pred_t_c = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=conditions, stage='eval')
            
            pred_l = (1+guidance)*pred_l_c - guidance*pred_l_u
            pred_t = (1+guidance)*pred_t_c - guidance*pred_t_u
            pred_x = (1+guidance)*pred_x_c - guidance*pred_x_u

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            # Denoised atom types
            fwd_atom_types = self.diffuse_atom_types(env_atom_types, t-1)
            t_t_minus_1 = fwd_mask * fwd_atom_types + (1 - fwd_mask) * t_t_minus_1

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }

        return traj[0], traj_stack
    
    @torch.no_grad()
    def conditional_sample(
        self,
        batch: Batch,
        diff_ratio: float = 1.0,
        step_lr: float = 1e-5,
        guidance: float = 3.0,
        targets: List[float] = [4.0, 2.0]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Generate a conditional sample trajectory guided by target property values.
        """
        batch_size = batch.num_graphs

        conditions = [] 
        for c,scaler in zip(targets,self.scaler):
            c = torch.tensor([[c]]*len(batch), dtype=torch.float32).view(-1,1).to(self.device)
            c = scaler.transform(c)
            conditions.append(c)

        conditions = torch.hstack(conditions)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,                  #initial data just noise at T=1000
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times    = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords'] #noisy version of frac coords at time step t
            l_t = traj[t]['lattices'] #noisy version of lattices at time step t
            t_t = traj[t]['atom_types'] #noisy version of types at time step t

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            
            _, pred_x_u, _ = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=None, stage='eval')
            _,pred_x_c, _  = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=conditions, stage='eval')

            pred_x = (1 + guidance) * pred_x_c - guidance * pred_x_u
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l_u, pred_x_u, pred_t_u = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=None, stage='eval')
            pred_l_c, pred_x_c, pred_t_c = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=conditions, stage='eval')
            
            pred_l = (1 + guidance) * pred_l_c - guidance * pred_l_u
            pred_t = (1 + guidance) * pred_t_c - guidance * pred_t_u
            pred_x = (1 + guidance) * pred_x_c - guidance * pred_x_u

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }

        return traj[0], traj_stack
        
    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5, random_state=42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        batch_size = batch.num_graphs
        l_T, x_T   = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        
        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,                  #initial data just noise at T=1000
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times    = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords'] #noisy version of frac coords at time step t
            l_t = traj[t]['lattices'] #noisy version of lattices at time step t
            t_t = traj[t]['atom_types'] #noisy version of types at time step t

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # Corrector (this only accounts for fractional coordinates (?))
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
        
            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=None, stage='eval')

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=None, stage='eval')

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }

        return traj[0], traj_stack
   
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict  = self(batch)
        loss_lattice = output_dict['loss_lattice']
        loss_coord   = output_dict['loss_coord']
        loss_type    = output_dict['loss_type']
        loss_cmpt    = output_dict['loss_cmpt']
        loss         = output_dict['loss']

        if loss.isnan() or loss.isinf():
            print(f"[rank {self.global_rank}] skipping batch {batch_idx}: loss={loss:.4f} "
                  f"(lattice={loss_lattice:.4f}, coord={loss_coord:.4f}, "
                  f"cmpt={loss_cmpt:.4f}, type={loss_type:.4f})")
            return torch.zeros(1, requires_grad=True, device=self.device)

        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'cmpt_loss': loss_cmpt,
            'type_loss': loss_type},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict    = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            #sync_dist_op="mean",
            batch_size=batch.num_graphs)
        
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(log_dict, batch_size=batch.num_graphs)
        return loss

    def compute_stats(self, output_dict, prefix):
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_cmpt = output_dict['loss_cmpt']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
            f'{prefix}_cmpt_loss': loss_cmpt
        }

        return log_dict, loss


class ConstrainedCSPDiffusion(CSPDiffusion):
    @torch.no_grad()
    def conditional_sample(
        self,
        batch,
        diff_ratio: float = 1.0,
        step_lr: float = 1e-5,
        guidance: float = 3.0,
        targets: List[float] = [4.0, 2.0],
        allowed_atomic_numbers: Optional[List[int]] = None,
        mask_neg_value: float = -1e4,
    ):
        batch_size = batch.num_graphs
        conditions = []
        for c, scaler in zip(targets, self.scaler):
            c = torch.tensor([[c]] * len(batch), dtype=torch.float32).view(-1, 1).to(self.device)
            c = scaler.transform(c)
            conditions.append(c)
        conditions = torch.hstack(conditions)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        allowed_mask = None
        if allowed_atomic_numbers is not None:
            allowed_mask = make_type_mask(allowed_atomic_numbers, MAX_ATOMIC_NUM, self.device)
            t_T = apply_type_mask(t_T, allowed_mask, mask_neg_value)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {self.beta_scheduler.timesteps: {
            'num_atoms': batch.num_atoms,
            'atom_types': t_T,
            'frac_coords': x_T % 1.,
            'lattices': l_T
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # --- Corrector ---
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            _, pred_x_u, _ = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=None, stage='eval')
            _, pred_x_c, _ = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch, condition=conditions, stage='eval')
            pred_x = ((1 + guidance) * pred_x_c - guidance * pred_x_u) * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t  # atom types untouched by corrector in the original code

            # --- Predictor ---
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * step_size) / (sigma_x ** 2))

            pred_l_u, pred_x_u, pred_t_u = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=None, stage='eval')
            pred_l_c, pred_x_c, pred_t_c = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, condition=conditions, stage='eval')

            pred_l = (1 + guidance) * pred_l_c - guidance * pred_l_u
            pred_t = (1 + guidance) * pred_t_c - guidance * pred_t_u
            pred_x = ((1 + guidance) * pred_x_c - guidance * pred_x_u) * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            # *** enforce the element constraint every step ***
            if allowed_mask is not None:
                t_t_minus_1 = apply_type_mask(t_t_minus_1, allowed_mask, mask_neg_value)

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': t_t_minus_1,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
        }
        return traj[0], traj_stack
