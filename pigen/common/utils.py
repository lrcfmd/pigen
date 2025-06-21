import os
import logging
import torch
import dotenv
from dataclasses import asdict
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import yaml

@rank_zero_only
def combine_and_save_to_yaml(dictionaries, file_path): 
    """define a function to combine dictionaries"""
    combined_dict = {}
    for dictionary in dictionaries:
        combined_dict.update(asdict(dictionary))

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w') as file:
        yaml.dump(combined_dict, file, default_flow_style=False)

def set_logger(logdir, name, level='INFO', console=False):
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = os.path.join(logdir, f'{name}.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('{asctime} | {levelname} | {message}', style = '{')
    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if console:
        chandler = logging.StreamHandler()
        chandler.setLevel(level)
        chandler.setFormatter(formatter)
        logger.addHandler(chandler)

    return logger

def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
