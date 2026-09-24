

## Setup environment
conda env create -f environment.yml
conda activate pigen

## Install package (editable mode)
pip install -e .

## Data used for training can be accessed at
https://huggingface.co/datasets/UoLiverpool/Alex_MP_20_M_LED/tree/main

## Testing
Basic tests can be run with pytest tests/ after setting up the conda environment with environment.yml. 
Most tests run in under 10 seconds.
