import argparse
import logging
import os
from datetime import datetime

from dataclasses import asdict
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader as GDataLoader
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

#from pigen.assets.diffusion_pi import CSPDiffusion
#from pigen.assets.diffusion_pi_cmptdiff import CSPDiffusion
from pigen.assets.diffusion_pi import CSPDiffusion
from pigen.assets.simple_dataset import SimpleCrystDataset
from pigen.common.utils import combine_and_save_to_yaml, set_logger
from pigen.settings import config

def update_config_with_args(args):
    if args.prop:
        config.data.prop = args.prop
        config.data.prop_weights = [1.0 for _ in args.prop]
    if args.data_name:
        config.data_name = args.data_name
    if args.gpus:
        config.gpus = args.gpus
    if args.random_state:
        config.random_state = args.random_state
    if args.ckpt_path:
        config.checkpoint.ckpt_path = args.ckpt_path
    if args.log is not None:
        config.log = args.log
    if args.p_cond:
        config.model.p_cond = args.p_cond
    config.experiment = args.experiment

def main():
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f'{config.PATHS.LOG_DIR}/{config.experiment}/{time_str}'

    logger = set_logger(log_dir, 'training', 'INFO')

    callbacks = [ModelCheckpoint(dirpath=log_dir),
                EarlyStopping(**asdict(config.earlystop))]
                # for restart from ckpt and resetting patience:
                #EarlyStopping(monitor='val_loss', patience=500, mode='min')]
                # for starting fresh:

    metric_logger = CSVLogger(log_dir, name=f'{config.experiment}')

    seed_everything(config.random_state)

    pl_trainer_params = asdict(config.trainer)

    from lightning_fabric.plugins.io import TorchCheckpointIO
    class LegacyCheckpointIO(TorchCheckpointIO):
        def load_checkpoint(self, path, map_location=None, weights_only=False):
            return super().load_checkpoint(path, map_location=map_location, weights_only=False)

    trainer = Trainer(
            default_root_dir='.',
            strategy=DDPStrategy(find_unused_parameters=True),
            logger=metric_logger,
            callbacks=callbacks,
            plugins=[LegacyCheckpointIO()],
            **pl_trainer_params
            )

    logger.info(f'Trainer initialized with params: \n{config.trainer}')
    # Save all config groups to YAML
    combine_and_save_to_yaml([
        config.data,
        config.scheduler,
        config.model,
        config.trainer,
        config.checkpoint
    ], log_dir+'/settings.yaml')


    train_df = pd.read_csv(f'{config.PATHS.DATA_DIR}/{config.data_name}/train.csv')
    val_df   = pd.read_csv(f'{config.PATHS.DATA_DIR}/{config.data_name}/val.csv')
    test_df  = pd.read_csv(f'{config.PATHS.DATA_DIR}/{config.data_name}/test.csv')


    logger.info(f'Data is read from {config.PATHS.DATA_DIR}')
    logger.debug(f'data_params config: {config.data}')

    prop = '_'.join(config.data.prop)

    train_dataset = SimpleCrystDataset(df=train_df,
                                        save_path=f'{config.PATHS.DATA_DIR}/{config.data_name}/train_ori_{prop}.pt',
                                        target_energy=True,
                                        gpus=config.gpus,
                                        **asdict(config.data))

    val_dataset   = SimpleCrystDataset(df=val_df,
                                        save_path=f'{config.PATHS.DATA_DIR}/{config.data_name}/val_ori_{prop}.pt',
                                        target_energy=True,
                                        gpus=config.gpus,
                                        **asdict(config.data))

    test_dataset  = SimpleCrystDataset(df=test_df,
                                        save_path=f'{config.PATHS.DATA_DIR}/{config.data_name}/test_ori_{prop}.pt',
                                        target_energy=True,
                                        gpus=config.gpus,
                                        **asdict(config.data))

    train_loader = GDataLoader(train_dataset, num_workers=config.data.preprocess_workers,  batch_size=config.data.batch_size, shuffle=False, drop_last=True, pin_memory=False)
    val_loader   = GDataLoader(val_dataset,   num_workers=config.data.preprocess_workers,  batch_size=config.data.batch_size, shuffle=False)
    test_loader  = GDataLoader(test_dataset,  num_workers=config.data.preprocess_workers,  batch_size=config.data.batch_size, shuffle=False)

    logger.info(f"DataLoaders are set with batch_size: {config.data.batch_size}")

    if len(config.data.prop) != len(config.data.prop_weights):
        logger.error(f"prop and prop_weights length mismatch: "
                     f"{len(config.data.prop)} vs {len(config.data.prop_weights)}")
        raise ValueError("prop and prop_weights must have the same length!")


    model = CSPDiffusion(**asdict(config.model), **asdict(config.data), **asdict(config.scheduler))
    logger.info(f'Model diffusion initialized with parameters: {config.model}')

    #Passing scalers to model
    model.scaler = train_dataset.scaler.copy()
    model.lattice_scaler = train_dataset.lattice_scaler.copy()

    logger.info(f'Fitting the model with scheduler: {config.scheduler}')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config.checkpoint.ckpt_path)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Run Conditional PIGEN model Training')
    parser.add_argument('--data_name', type=str, default='Alex_MP_20_M_LED',  help='Name of the dataset')
    parser.add_argument('--prop', type=str, nargs='+', default=['entropy_sum', 'target_energy'], help='Conditioning properties')
    parser.add_argument('--p_cond', type=float, default=0.4, help='Probability of conditioning for CFG')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint to load')
    parser.add_argument('--log', action='store_true', default=False, help='Enable metric logging')
    parser.add_argument('--gpus', type=int, default=1,  help='Number of gpus')
    parser.add_argument('--random_state', type=int, default=42,  help='Random state for reproducibility')
    parser.add_argument('--experiment', type=str, default='dummy',  help='Folder to place the ckpt')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    update_config_with_args(args)
    main()
