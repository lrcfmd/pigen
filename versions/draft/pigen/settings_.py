from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
LOG_DIR = PROJECT_ROOT / 'outputs' / 'logs'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

data_name    = 'full_data'
random_state = 1234

'''Dataset Settings'''
data_params = {'primitive' : True, 
                'niggli' : True,
                'preprocess_workers': 4,
                'lattice_scale_method' : 'scale_length',
                'tolerance': 0.1,
                'prop' : ['CH'], # Enegy above convex hull (CH)
                'cat_prop': False, #Categorical property, no scaling 
                'prop_weights': [1.], #between 0 and 1
                'use_space_group' : False,
                'use_pos_index': False,
                'batch_size': 24}

'''Scheduler Settings'''
scheduler_hparams = {'scheduler_mode' : 'cosine'}

'''Model settings'''
model_hparams = {'latent_dim' : 0,
                'p_cond' : 0.4,
                'cost_coord' : 1.,
                'cost_lattice' : 1.,
                'time_dim' : 256,
                'cost_type' : 20.,
                'cost_cmpt' : 1.0,
                'max_neighbors' : 20,
                'radius' : 7.,
                'timesteps' : 1000,
                'optim': {'use_lr_scheduler':True,
                           'lr': 1e-3,
                            'weight_decay': 0,
                            'betas': [0.9, 0.999],
                            'eps': 1e-8,
                            'lr_patience': 30,
                            'lr_factor': 0.6,
                            'min_lr': 1e-4}}


'''PL Training settings'''
pl_trainer_params = {
    'accelerator': 'gpu',
    'devices': 4,
    'num_nodes': 2,
    'precision': 32,
    'max_epochs': 1000,
    'accumulate_grad_batches': 4,
    'gradient_clip_val': 0.5,
    'gradient_clip_algorithm': 'value',
    'fast_dev_run': False,
    'num_sanity_val_steps': 2,
    'deterministic': True
}
#Model Checkpoint
checkpoint_params = {'monitor'   : 'val_coord_loss',
                    'save_top_k' : 1,
                    'verbose'    : False,
                    'mode'       : 'min'}

#Early Stopping
earlystop_params = {'monitor' : 'val_loss',
                    'patience' : 120,
                    'verbose' : False,
                    'mode' : 'min'}

'''Logger settings'''
log = False
wandb_params = {'project': 'pigen',
                'name': f"ALEX_MP20_MLED",
                'entity': 'andrij',
                'log_model': True,
                'mode': 'online'}

wandb_watch = {'log': 'all', 'log_freq': 100}

ckpt= None
