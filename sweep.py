import wandb
import yaml
from train import train
with open('sweep.yaml', 'r') as file:
    config = yaml.safe_load(file) 

sweep_id = wandb.sweep(sweep=config, project="unet_ssl")
wandb.agent(sweep_id, function=train, count=60)
