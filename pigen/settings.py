from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / 'data')
    LOG_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / 'log')
    CHECKPOINT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / 'checkpoints')

@dataclass
class DataParams:
    primitive: bool = True
    niggli: bool = True
    preprocess_workers: int = 4
    lattice_scale_method: str = 'scale_length'
    tolerance: float = 0.1
    prop: Optional[List[str]] = field(default_factory=lambda: ['target_energy'])
    cat_prop: bool = False
    prop_weights: Optional[List[float]] = field(default_factory=lambda: [1.0])
    use_space_group: bool = False
    use_pos_index: bool = False
    batch_size: int = 24

@dataclass
class SchedulerParams:
    scheduler_mode: str = 'cosine'

@dataclass
class TrainerParams:
    accelerator: str = 'gpu'
    devices: int = 4
    num_nodes: int = 4
    fast_dev_run: bool = False
    precision: int = 32
    gradient_clip_val: float = 0.5
    gradient_clip_algorithm: str = 'value'
    max_epochs: int = 3000
    num_sanity_val_steps: int = 2
    accumulate_grad_batches: int = 4
    deterministic: bool = True


@dataclass
class OptimizerParams:
    use_lr_scheduler: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    lr_patience: int = 30
    lr_factor: float = 0.6
    min_lr: float = 1e-4


@dataclass
class ModelParams:
    latent_dim: int = 0
    p_cond: float = 0.4
    cost_coord: float = 1.0
    cost_lattice: float = 1.0
    time_dim: int = 256
    cost_type: float = 20.0
    cost_cmpt: float = 1.0
    max_neighbors: int = 20
    radius: float = 7.0
    timesteps: int = 1000
    optim: OptimizerParams = field(default_factory=OptimizerParams)

@dataclass
class CheckpointParams:
    ckpt_path: Optional[str] = None
    monitor: str = 'val_coord_loss'
    save_top_k: str = 1
    verbose: bool = False
    mode: str ='min'

@dataclass
class EarlyStopParams:
    monitor: str = 'val_loss'
    patience: int = 1000
    verbose: bool = False
    mode: str = 'min'

@dataclass
class AppConfig:
    PATHS: Paths = field(default_factory=Paths)
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    trainer: TrainerParams = field(default_factory=TrainerParams)
    scheduler: SchedulerParams = field(default_factory=SchedulerParams)
    checkpoint: CheckpointParams = field(default_factory=CheckpointParams)
    earlystop: EarlyStopParams = field(default_factory=EarlyStopParams)
    optimizer: OptimizerParams = field(default_factory=OptimizerParams)
    data_name: str = 'full_data'
    random_state: int = 1234
    log: bool = True
    gpus: Optional[int] = None


# Create a singleton-ish config object
config = AppConfig()
config.model.optim = config.optimizer
