"""Test all dependencies"""
from assets.simple_dataset import SimpleCrystDataset
import torch
from torch_geometric.loader import DataLoader as GDataLoader
from pigen.common.utils import combine_and_save_to_yaml
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pigen.assets.diffusion import CSPDiffusion
import pandas as pd
from datetime import datetime
import numpy as np
from pigen.settings import *
import argparse

print(torch.cuda.is_available())
print(torch.__version__)
