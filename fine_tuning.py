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
from datasets import HAR_dataset_fine

import yaml
config_path = "config_new.yaml"  if len(sys.argv) < 2 else sys.argv[-1]
with open(config_path, 'r') as file:
	config = yaml.safe_load(file) 
from torch.utils.data import DataLoader
import wandb

'''dataset = HAR_dataset_fine(transform = transforms.Compose([
			  transforms.ToTensor()
		  ]))'''
#creates the train, val, and test datasets
dataset=HAR_dataset_fine()
train_dataset = torch.utils.data.Subset(dataset, range(dataset.train_sz))
val_dataset = torch.utils.data.Subset(dataset, range(dataset.train_sz, dataset.train_sz+dataset.val_sz))
test_dataset = torch.utils.data.Subset(dataset, range(dataset.train_sz+dataset.val_sz, dataset.train_sz+dataset.val_sz+dataset.test_sz))
print("total=", len(dataset), "train=", len(train_dataset), "val=", len(val_dataset), "test=", len(test_dataset))

#used to generate weights for the WeightedRandomSampling used in the dataloader
def make_weights_for_balanced_classes(data, nclasses):
	freqs = [0]*nclasses
	for (reg, activity) in data:
		freqs[activity] += 1
	weight_per_class = [0.]*nclasses
	N = float(sum(freqs))
	for i in range(1, nclasses):
		weight_per_class[i] = N/float(freqs[i])
	weight = [0]*len(data)
	for idx, val in enumerate(data):
		weight[idx] = weight_per_class[val[1]]
	return weight

#loads the pretrained model from a saved checkpoint
unet_pretrained = UNet(n_classes = 3, depth = config['depth'], wf=2, padding = True)

checkpoint = torch.load(config['checkpoint_path']) #replace this with the model path
unet_pretrained.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
print("Epoch =", epoch)
loss = checkpoint['loss']
print("Loss =", loss)

checkpoints_dir = f"temp" #where the final model will be stored

def train():
	#initializing directories and wandb
	wandb.init(project="unet_ssl", entity="jli505", config=config, reinit=True)
	cfg = wandb.config
	global checkpoints_dir
	checkpoints_dir = f"{wandb.run.dir}/checkpoints"
	os.makedirs(checkpoints_dir, exist_ok = True)

	test_dir = f"{cfg.data_dir}/{cfg.val_dir}/{cfg.noisy_dir}"

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print('device: ', device)


	script_time = time.time()

	def q(text = ''):
		print('> {}'.format(text))
		sys.exit()

	data_dir = cfg.data_dir
	train_dir = cfg.train_dir
	val_dir = cfg.val_dir
		
	models_dir = cfg.models_dir
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	losses_dir = cfg.losses_dir
	if not os.path.exists(losses_dir):
		os.mkdir(losses_dir)

	#useless
	def count_parameters(model):
		num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
		return num_parameters/1e6 # in terms of millions
	#useless
	def plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
		fig = plt.figure(figsize=(16,16))
		fig.suptitle('loss trends', fontsize=20)
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)

		ax1.title.set_text('epoch train loss VS #epochs')
		ax1.set_xlabel('#epochs')
		ax1.set_ylabel('epoch train loss')
		ax1.plot(train_epoch_loss)
		
		ax2.title.set_text('epoch val loss VS #epochs')
		ax2.set_xlabel('#epochs')
		ax2.set_ylabel('epoch val loss')
		ax2.plot(val_epoch_loss)
	 
		ax3.title.set_text('batch train loss VS #batches')
		ax3.set_xlabel('#batches')
		ax3.set_ylabel('batch train loss')
		ax3.plot(running_train_loss)

		ax4.title.set_text('batch val loss VS #batches')
		ax4.set_xlabel('#batches')
		ax4.set_ylabel('batch val loss')
		ax4.plot(running_val_loss)
		
		plt.savefig(os.path.join(losses_dir,'losses_{}.png'.format(str(epoch + 1).zfill(2))))

	transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

	#load the training dataset with a weighted sampler
	batch_size = cfg.batch_size
	weights = make_weights_for_balanced_classes(train_dataset, 8)
	weights = torch.DoubleTensor(weights)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler = sampler)
	
	#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

	print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
	print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))

	#defines the model used in this fine tuning task
	model = UNet_Fine(unet_pretrained) # try decreasing the depth value if there is a memory error
	model.to(device)
	resume = cfg.resume

	#whether or not I already started training this model - I always set resume to false in the config file
	if not resume:
		print('\nfrom scratch')
		train_epoch_loss = []
		val_epoch_loss = []
		running_train_loss = []
		running_val_loss = []
		epochs_till_now = 0
	else:
		ckpt_path = os.path.join(models_dir, cfg.ckpt)
		ckpt = torch.load(ckpt_path)
		print(f'\nckpt loaded: {ckpt_path}')
		model_state_dict = ckpt['model_state_dict']
		model.load_state_dict(model_state_dict)
		model.to(device)
		losses = ckpt['losses']
		running_train_loss = losses['running_train_loss']
		running_val_loss = losses['running_val_loss']
		train_epoch_loss = losses['train_epoch_loss']
		val_epoch_loss = losses['val_epoch_loss']
		epochs_till_now = ckpt['epochs_till_now']

	#learning rate, optimizer, loss function
	lr = cfg.lr
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
	loss_fn = nn.CrossEntropyLoss()

	log_interval = cfg.log_interval
	epochs = cfg.epochs

	#training and validation - keep track of the minimum validation loss so that the model updates whenever it achieves a smaller validation loss
	min_val_loss = 1000000000
	for epoch in range(epochs_till_now, epochs_till_now+epochs):
		epoch_train_start_time = time.time()
		model.train()
		tr_mean = 0
		tr_mean_loss = 0
		#training loop
		for batch_idx, (imgs, activity) in enumerate(train_loader):
			batch_start_time = time.time()
			imgs = imgs.to(device)
			activity = activity.to(device)
			tr_mean += imgs.mean(dim=0, keepdim=True)
			optimizer.zero_grad()
			out = model(imgs)
			loss = loss_fn(out, activity)
			running_train_loss.append(loss.item())
			loss.backward()
			optimizer.step()

			if (batch_idx + 1)%log_interval == 0:
				batch_time = time.time() - batch_start_time
				m,s = divmod(batch_time, 60)

		tr_mean /= len(train_loader)
		tr_mean_loss /= len(train_loader)
		mean_train_loss = np.array(running_train_loss).mean()
		train_epoch_loss.append(mean_train_loss)
		wandb.log({"train_loss":mean_train_loss})

		#used to keep track of how long it takes the model to train
		epoch_train_time = time.time() - epoch_train_start_time
		m,s = divmod(epoch_train_time, 60)
		h,m = divmod(m, 60)

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
		running_train_loss, running_val_loss = [], []
		#print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))
	total_script_time = time.time() - script_time
	m, s = divmod(total_script_time, 60)
	h, m = divmod(m, 60)
	print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
	print('\nFin.')

train()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#testing the accuracy of my model
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)
unet_pretrained = UNet(n_classes = 3, depth = config['depth'], wf=2, padding = True)
model = UNet_Fine(unet_pretrained)

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
