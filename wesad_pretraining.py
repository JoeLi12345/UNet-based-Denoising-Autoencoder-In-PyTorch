import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg') # for servers not supporting display

sys.path.insert(0,'..')

# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from mlp import MLP
from unet import UNet
from datasets import WESADDataset

import yaml
config_path = "config_wesad_pretraining.yaml"
with open(config_path, 'r') as file:
	config = yaml.safe_load(file) 
from torch.utils.data import DataLoader
import wandb


def init_wandb():
	wandb.init(project="unet_ssl", entity="jli505", config=config, reinit=True)
	global checkpoints_dir
	checkpoints_dir = f"{wandb.run.dir}/checkpoints"
	os.makedirs(checkpoints_dir, exist_ok = True)
	global cfg
	cfg = wandb.config
	print("cfg.lr=", cfg.lr)
	return checkpoints_dir

def train(subject):
	torch.manual_seed(1347)
	train_dataset = WESADDataset(split="train", subject=subject)
	val_dataset = WESADDataset(split="validation", subject=subject)
	print("train, val = ", len(train_dataset), len(val_dataset))
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print('device: ', device)

	script_time = time.time()
	#create data loaders for train, val, test
	transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
	batch_size = cfg.batch_size
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

	#defining the UNET model here
	print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
	print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))
	model = UNet(in_channels=train_dataset.in_channels, n_classes = train_dataset.in_channels, depth = cfg.depth, wf=2, padding = True) # try decreasing the depth value if there is a memory error
	model.to(device)

	#learning rate, optimizer, loss function
	lr = cfg.lr
	optimizer = optim.AdamW(model.parameters(), lr = lr)
	loss_fn = nn.MSELoss()

	train_epoch_loss, val_epoch_loss = [], []
	epochs_till_now = 0
	epochs = cfg.epochs
	min_val_loss = 1000000000

	patience = 30
	current_repeat = 0
	eps = 0.00001
	#epochs for training and validation
	for epoch in range(epochs_till_now, epochs_till_now+epochs):
		epoch_train_start_time = time.time()
		model.train()
		running_train_loss, running_val_loss = [], []
		print("Epoch {}:\n".format(epoch))
		#training
		for batch_idx, (imgs, noisy_imgs) in enumerate(train_loader):
			batch_start_time = time.time()
			imgs = imgs.to(device)
			noisy_imgs = noisy_imgs.to(device)
			optimizer.zero_grad()
			out = model(noisy_imgs)
			loss = loss_fn(out, imgs)
			running_train_loss.append(loss.item())
			loss.backward()
			optimizer.step()

		#keeping track of losses to store in wandb
		mean_train_loss = np.array(running_train_loss).mean()
		train_epoch_loss.append(mean_train_loss)
		wandb.log({"train_loss":mean_train_loss})

		epoch_train_time = time.time() - epoch_train_start_time
		m,s = divmod(epoch_train_time, 60)
		h,m = divmod(m, 60)
		print("Train time: {} hrs {} mins {} secs".format(int(h), int(m), int(s)))
		print("Train loss: {}".format(mean_train_loss))
		
		#begin validation
		epoch_val_start_time = time.time()
		model.eval()
		with torch.no_grad():
			for batch_idx, (imgs, noisy_imgs) in enumerate(val_loader):

				imgs = imgs.to(device)
				noisy_imgs = noisy_imgs.to(device)

				out = model(noisy_imgs)
				loss = loss_fn(out, imgs)

				running_val_loss.append(loss.item())

		#if the new validation loss is smaller then update the model
		mean_val_loss = np.array(running_val_loss).mean()
		if mean_val_loss < min_val_loss:
			torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, f"{checkpoints_dir}/best.pt") #replace path
		if min_val_loss-mean_val_loss < eps:
			current_repeat += 1
		else:
			current_repeat = 0
		if current_repeat == patience:
			break
		min_val_loss = min(min_val_loss, mean_val_loss)	
		val_epoch_loss.append(mean_val_loss)
		wandb.log({"val_loss":mean_val_loss})

		epoch_val_time = time.time() - epoch_val_start_time
		m,s = divmod(epoch_val_time, 60)
		h,m = divmod(m, 60)
		print("Val time: {} hrs {} mins {} secs".format(int(h), int(m), int(s)))
		print("Val loss: {}".format(mean_val_loss))
	total_script_time = time.time() - script_time
	m, s = divmod(total_script_time, 60)
	h, m = divmod(m, 60)
	print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
	print('\nFin.')


init_wandb()
train(7)
