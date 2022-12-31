import numpy as np
import os, glob, cv2, sys
import torch
import random
from pandas import read_csv
import pandas as pd
import pickle
from torchvision import transforms

#Data Loader for pretraining task
class HAR_dataset(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		self.transform = transform
		self.load_dataset()
		self.format_data()

	#load the files from HAR directory
	def load_dataset(self):
		subjects = list()
		directory = '../HAR/'
		for name in os.listdir(directory):
			filename = directory + '/' + name
			if not filename.endswith('.csv'):
				continue
			df = read_csv(filename, header = None)
			values = df.values[:,1:]
			subjects.append(values)
		self.subject = subjects[0]
	
	#create "self.data" such that the data is in the order of train, val, and test
	#create windows with 64 data points each
	def format_data(self):
		print(len(self.subject))
		dx = 64
		cur = 0
		inc = 1
		num_activities = 8
		self.data = []
		category = []
		for i in range(num_activities):
			category.append([])
		#partition into windows of length 64
		while (cur+dx-1 < len(self.subject)):
			value = round(self.subject[cur][3])
			add = True
			x_vals = []
			y_vals = []
			z_vals = []
			for i in range(cur, cur+dx):
				x_vals.append(self.subject[i][0])
				y_vals.append(self.subject[i][1])
				z_vals.append(self.subject[i][2])
				if (self.subject[i][3] != value):
					add = False
					break
			if (add):
				category[value].append([x_vals, y_vals, z_vals])
			cur += inc
		#divide the data between train, val, and test according to the 70-15-15 split
		train_data, val_data, test_data = [], [], []
		for i in range(num_activities):
			if len(category[i]) == 0:
				continue
			train_cnt = int(0.7*len(category[i]))
			val_cnt = int(0.15*len(category[i]))
			test_cnt = len(category[i])-train_cnt-val_cnt
			for j in range(train_cnt):
				train_data.append(category[i][j])
			for j in range(train_cnt, train_cnt+val_cnt):
				val_data.append(category[i][j])
			for j in range(train_cnt+val_cnt, train_cnt+val_cnt+test_cnt):
				test_data.append(category[i][j])
		#add the train, val, test back to self.data --> now you know exactly where the train, val, and test data points are when you access "self.data" outside of this class
		self.train_sz = len(train_data)
		self.val_sz = len(val_data)
		self.test_sz = len(test_data)
		for i in train_data:
			self.data.append(i)
		for i in val_data:
			self.data.append(i)
		for i in test_data:
			self.data.append(i)
			
	def __getitem__(self, index):
		reg = self.data[index]
		noised = reg
		#implement masking by randomly selecting some data points to mask out
		subset_indices = random.sample(range(len(reg)), int(random.uniform(0, 0.15)*len(reg)))
		for i in subset_indices:
			noised[i] = [-1, -1, -1]
		reg = torch.tensor(reg, dtype=torch.float)
		noised = torch.tensor(noised, dtype=torch.float)
		if self.transform:
			reg = self.transform(reg)
			noised = self.transform(noised)
		return reg, noised

	def __len__(self):
		return len(self.data)

#data loader for the transfer learning task
class HAR_dataset_fine(HAR_dataset):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.load_dataset()
		self.format_data()

	#load dataset from HAR
	def load_dataset(self):
		subjects = list()
		directory = '../HAR/'
		for name in os.listdir(directory):
			filename = directory + '/' + name
			if not filename.endswith('.csv'):
				continue
			df = read_csv(filename, header = None)
			values = df.values[:,1:]
			subjects.append(values)
		self.subject = subjects[0]
	
	#exactly the same as the parent method, except now we maintain the activity
	def format_data(self):
		print(len(self.subject))
		dx = 64
		cur = 0
		inc = 1
		num_activities = 8
		self.data = []
		self.labels = []
		category = []
		for i in range(num_activities):
			category.append([])
		while (cur+dx-1 < len(self.subject)):
			value = round(self.subject[cur][3])
			add = True
			x_vals = []
			y_vals = []
			z_vals = []
			for i in range(cur, cur+dx):
				x_vals.append(self.subject[i][0])
				y_vals.append(self.subject[i][1])
				z_vals.append(self.subject[i][2])
				if (self.subject[i][3] != value):
					add = False
					break
			if (add):
				category[value].append(([x_vals, y_vals, z_vals], value))
			cur += inc
		train_data, val_data, test_data = [], [], []
		for i in range(num_activities):
			if len(category[i]) == 0:
				continue
			train_cnt = int(0.7*len(category[i]))
			val_cnt = int(0.15*len(category[i]))
			test_cnt = len(category[i])-train_cnt-val_cnt
			for j in range(train_cnt):
				train_data.append(category[i][j])
			for j in range(train_cnt, train_cnt+val_cnt):
				val_data.append(category[i][j])
			for j in range(train_cnt+val_cnt, train_cnt+val_cnt+test_cnt):
				test_data.append(category[i][j])
		self.train_sz = len(train_data)
		self.val_sz = len(val_data)
		self.test_sz = len(test_data)
		for i in train_data:
			self.data.append(i[0])
			self.labels.append(i[1])
		for i in val_data:
			self.data.append(i[0])
			self.labels.append(i[1])
		for i in test_data:
			self.data.append(i[0])
			self.labels.append(i[1])

	#returns the data and its label
	def __getitem__(self, index):
		reg = self.data[index]
		reg = torch.tensor(reg, dtype=torch.float)
		activity = self.labels[index]
		activity = torch.tensor(activity, dtype=torch.int64)
		if self.transform:
			reg = self.transform(reg)
		return reg, activity

	def __len__(self):
		return len(self.data)

class WESADDataset(torch.utils.data.Dataset):
	def __init__(self, split, subject, transform=None):
		self.split = split
		self.subject = subject
		self.transform = transform
		self.load_dataset()
		self.format_data()

	#load the files from HAR directory

	def load_dataset(self):
		vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
		x = vals[self.subject]
		df = pd.read_pickle('../WESAD/S'+str(x)+'/S'+str(x)+'.pkl')
		self.signals = []
		self.labels = []
		self.in_channels = 0
		for keys in df['signal']['chest']:
			self.in_channels += df['signal']['chest'][keys].shape[1]
		for keys in df['signal']['chest']:
			x = df['signal']['chest'][keys]
			for i in range(x.shape[1]):
				arr = []
				for j in range(x.shape[0]):
					arr.append(x[j][i])
				self.signals.append(arr)
		for x in df['label']:
			self.labels.append(x)
	
	#create "self.data" such that the data is in the order of train, val, and test
	#create windows with 64 data points each
	def format_data(self):
		dx, cur, inc, num_activities = 256, 0, 128, 3
		self.data, self.activities, category = [], [], []
		for i in range(num_activities):
			category.append([])
		print(len(self.signals[0]), len(self.labels))
		#partition into windows of length dx
		while (cur+dx-1 < len(self.signals[0])):
			value = self.labels[cur]
			add = True
			for i in range(cur, cur+dx):
				if self.labels[i] != value:
					add = False
					break
			if add and value >= 1 and value <= 3:
				window = []
				for i in range(self.in_channels):
					window.append(self.signals[i][cur:cur+dx])
				category[value-1].append(window)
			cur += inc
		#70, 15, 15 split for training, validation, testing
		for i in range(len(category)):
			n = len(category[i])
			partition = [0, int(0.7*n), int(0.85*n), n]
			partition_index = -1
			if self.split == "train":
				partition_index = 0
			elif self.split == "validation":
				partition_index = 1
			elif self.split == "test":
				partition_index = 2
			else:
				print("invalid train/val/test arg")
				exit()
			for j in range(partition[partition_index], partition[partition_index+1]):
				self.data.append(category[i][j])
				self.activities.append(i)
		self.mean = np.expand_dims(np.array(self.data).mean(axis=(0, 2)), -1)
		self.std = np.expand_dims(np.array(self.data).std(axis=(0, 2)), -1)
		if self.transform is None:
			self.transform = lambda x: (x-self.mean)/self.std
		

	def __getitem__(self, index):
		reg = self.data[index]
		noised = reg
		#implement masking by randomly selecting some data points to mask out
		subset_indices = random.sample(range(len(reg)), int(random.uniform(0, 0.15)*len(reg)))
		for i in subset_indices:
			for j in range(len(noised)):
				noised[j][i] = -1
		reg = torch.tensor(reg, dtype=torch.float)
		noised = torch.tensor(noised, dtype=torch.float)
		if self.transform:
			reg = self.transform(reg).float()
			noised = self.transform(noised).float()
		return reg, noised

	def __len__(self):
		return len(self.data)

#data loader for the transfer learning task
class WESADDatasetFine(WESADDataset):
	def __init__(self, split, subject, transform=None):
		super().__init__(split, subject, transform)

	def __getitem__(self, index):
		X = self.data[index]
		X = torch.tensor(X, dtype=torch.float)
		if self.transform:
			X = self.transform(X).float()
		y = self.activities[index]
		y = torch.tensor(y, dtype=torch.int64)
		return X, y
