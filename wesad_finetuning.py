import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
plt.switch_backend('agg') # for servers not supporting display

sys.path.insert(0,'..')

# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from mlp import MLP
from unet import UNet
from unet_nn import UNet_Fine
from datasets import WESADDatasetFine

import yaml
config_path = "config_wesad_finetuning.yaml"  if len(sys.argv) < 2 else sys.argv[-1]
with open(config_path, 'r') as file:
	config = yaml.safe_load(file) 
from torch.utils.data import DataLoader
import wandb

def init_wandb():
	wandb.init(project="unet_ssl", entity="jli505", config=config, reinit=True)
	global cfg
	cfg = wandb.config
	global checkpoints_dir
	checkpoints_dir = f"{wandb.run.dir}/checkpoints"
	os.makedirs(checkpoints_dir, exist_ok = True)

#loads the pretrained model from a saved checkpoint

def train(subject):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print('device: ', device)
	train_dataset = WESADDatasetFine(split="train", subject=subject)
	val_dataset = WESADDatasetFine(split="validation", subject=subject)
	script_time = time.time()
	#load the training dataset with a weighted sampler
	batch_size = cfg.batch_size
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

	print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
	print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))


	unet_pretrained = UNet(in_channels=train_dataset.in_channels, n_classes = train_dataset.in_channels, depth = config['depth'], wf=2, padding = True)
	checkpoint = torch.load(config['checkpoint_path']) #replace this with the model path
	unet_pretrained.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch']
	print("Epoch =", epoch)
	loss = checkpoint['loss']
	print("Loss =", loss)

	#defines the model used in this fine tuning task
	model = UNet_Fine(unet_pretrained, num_classes=3, window_length=256) # try decreasing the depth value if there is a memory error
	model.to(device)

	#learning rate, optimizer, loss function
	lr = cfg.lr
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
	loss_fn = nn.CrossEntropyLoss()
	epochs = cfg.epochs

	#training and validation - keep track of the minimum validation loss so that the model updates whenever it achieves a smaller validation loss
	min_val_loss = 1000000000
	train_epoch_loss, val_epoch_loss = [], []
	epochs_till_now = 0
	for epoch in range(epochs_till_now, epochs_till_now+epochs):
		print("Epoch {}:\n".format(epoch))
		running_train_loss, running_val_loss = [], []
		epoch_train_start_time = time.time()
		model.train()
		#training loop
		for batch_idx, (imgs, activity) in enumerate(train_loader):
			batch_start_time = time.time()
			imgs = imgs.to(device)
			activity = activity.to(device)
			optimizer.zero_grad()
			out = model(imgs)
			loss = loss_fn(out, activity)
			running_train_loss.append(loss.item())
			loss.backward()
			optimizer.step()
		mean_train_loss = np.array(running_train_loss).mean()
		train_epoch_loss.append(mean_train_loss)
		wandb.log({"train_loss":mean_train_loss})
		#used to keep track of how long it takes the model to train
		epoch_train_time = time.time() - epoch_train_start_time
		m,s = divmod(epoch_train_time, 60)
		h,m = divmod(m, 60)
		print("train time: {} hrs {} mins {} secs".format(int(h), int(m), int(s)))
		print("train loss: {}".format(mean_train_loss))

		#validation loop
		epoch_val_start_time = time.time()
		model.eval()
		with torch.no_grad():
			for batch_idx, (imgs, activity) in enumerate(val_loader):

				imgs = imgs.to(device)
				activity = activity.to(device)
				
				out = model(imgs)
				loss = loss_fn(out, activity)
				

				running_val_loss.append(loss.item())

		#logging the losses on wandb
		mean_val_loss = np.array(running_val_loss).mean()
		if mean_val_loss < min_val_loss:
			torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss
			}, f"{checkpoints_dir}/best.pt") #replace path
		min_val_loss = min(min_val_loss, mean_val_loss)	

		val_epoch_loss.append(mean_val_loss)
		wandb.log({"val_loss":mean_val_loss})

		epoch_val_time = time.time() - epoch_val_start_time
		m,s = divmod(epoch_val_time, 60)
		h,m = divmod(m, 60)
		print("val time: {} hrs {} mins {} secs".format(int(h), int(m), int(s)))
		print("val loss: {}".format(mean_val_loss))
	total_script_time = time.time() - script_time
	m, s = divmod(total_script_time, 60)
	h, m = divmod(m, 60)
	print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
	print('\nFin.')

def test(subject):
	test_dataset = WESADDatasetFine(split="test", subject=subject)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)
	unet_pretrained = UNet(in_channels=test_dataset.in_channels, n_classes = test_dataset.in_channels, depth = config['depth'], wf=2, padding = True)
	model = UNet_Fine(unet_pretrained, num_classes=3, window_length=256)

	checkpoint = torch.load(f"{checkpoints_dir}/best.pt") #replace this with the model path
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)

	epoch = checkpoint['epoch']
	print("Epoch =", epoch)
	loss = checkpoint['loss']
	print("Loss =", loss)

	model.eval()

	correct = 0
	total = 0

	loss_fn = nn.CrossEntropyLoss()
	with torch.no_grad():
		running_test_loss = []
		for batch_idx, (imgs, activity) in enumerate(test_loader):

			imgs = imgs.to(device)
			activity = activity.to(device)
			

			out = model(imgs)
			loss = loss_fn(out, activity)
			running_test_loss.append(loss.item())
			print(out)
			total += activity.size(0)
			correct += (out.argmax(dim=1) == activity).sum().item()
	wandb.log({"test_loss": np.array(running_test_loss).mean()})

	print(f"Accuracy={correct/total*100}%")

init_wandb()
train(0)
test(0)
