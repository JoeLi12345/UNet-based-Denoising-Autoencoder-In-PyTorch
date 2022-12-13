import numpy as np
import os, glob, cv2, sys
import torch
import random
from pandas import read_csv

class HAR_dataset(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		self.transform = transform
		self.load_dataset()
		self.format_data()

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
	#sliding window data preparation
	
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
		train_data, val_data, test_data = [], [], []
		for i in range(num_activities):
			if len(category[i]) == 0:
				continue
			#print(i, ":", len(category[i]))
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
			self.data.append(i)
		for i in val_data:
			self.data.append(i)
		for i in test_data:
			self.data.append(i)
			
	
	def __getitem__(self, index):
		reg = self.data[index]
		noised = reg
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


class HAR_dataset_fine(HAR_dataset):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.load_dataset()
		self.format_data()

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
	#sliding window data preparation
	
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
			#print(i, ":", len(category[i]))
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
			
	def __getitem__(self, index):
		reg = self.data[index]
		reg = torch.tensor(reg, dtype=torch.float)
		activity = self.labels[index]
		#print("act:\n", activity)
		activity = torch.tensor(activity, dtype=torch.int64)
		#print("reg:\n", reg)
		#print("act:\n", activity)
		if self.transform:
			reg = self.transform(reg)
		return reg, activity

	def __len__(self):
		return len(self.data)
