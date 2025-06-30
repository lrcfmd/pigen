import pytest

def test_imports_and_versions():
    try:
        import torch
        import pandas as pd
        import numpy as np
        import pytorch_lightning as pl
        from torch_geometric.loader import DataLoader
        from pigen.assets.simple_dataset import SimpleCrystDataset
        from pigen.assets.diffusion_pi import CSPDiffusion
        from pigen.common.utils import combine_and_save_to_yaml
        from pigen.settings import config

    except ImportError as e:
        pytest.fail(f"Missing dependency or broken import: {e}")
