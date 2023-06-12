import wandb
import yaml
import subject_11_special
with open('sweep.yaml', 'r') as file:
    config = yaml.safe_load(file) 

print("SWEEP_INIT")
sweep_id = wandb.sweep(sweep=config, project="unet_ssl")
print("SWEEP BEGIN")
wandb.agent(sweep_id, function=subject_11_special.ssl, count=10)
