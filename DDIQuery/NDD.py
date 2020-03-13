import numpy as np
import scipy.io as io
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from batch import *
import sys
import matplotlib.pyplot as plt

folds = 10
weight_decay = 2e-3
learning_rate = 1.2e-3
epoch = 80
criterion = nn.BCELoss()

class NDD(nn.Module):
	def __init__(self, onehot_size):
		super(NDD, self).__init__()
		self.fc1 = nn.Linear(onehot_size * 2, 300, bias=True)
		self.fc2 = nn.Linear(300, 400, bias=True)
		self.fc3 = nn.Linear(400, 1, bias=True)

	def forward(self, x):
		activation = F.relu
		s = nn.Sigmoid()
		r1 = activation(self.fc1(x))
		r2 = activation(self.fc2(r1))
		r3 = s(self.fc3(r2))
		return r3

if __name__ == '__main__':
	np.random.seed()
	# batch = loadNDDdata()
	auc1s = np.zeros(folds)
	aupr1s = np.zeros(folds)
	auc3s = np.zeros(folds)
	aupr3s = np.zeros(folds)
	auc13s = np.zeros(folds)
	aupr13s = np.zeros(folds)
	
	# for index, repos in enumerate(batch.repos):
	train_repos, test_repos = loadNDDdata()
	model = NDD(train_repos.onehot_size)

	optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)


	train_input, train_output_onehot = train_repos.miniBatch(568)

	for i in range(epoch):

		optimizer.zero_grad()
		pred = model(train_input)
		loss = criterion(pred.flatten(), train_output_onehot.flatten())
		loss.backward()
		optimizer.step()

		train_repos.reset()

	
	# print(test_input.shape)
	# print(test_repos.input_feat.shape)
	test_input, test_output_onehot = test_repos.miniBatch(568)

	# print(test_output_onehot.shape)
	pred = model(test_input)
	out = pred.data.numpy()
	labels = test_output_onehot.data.numpy()
	# print(labels)

	auc = metrics.roc_auc_score(labels.flatten(), out.flatten())
	aupr = metrics.average_precision_score(labels.flatten(), out.flatten())
	print('Overall', auc, aupr)
	# print(np.where(labels[tr, :][:, te]!=labels[te, :][:, tr].transpose()))


# 	tr = test_repos.input_ids
# 	te = test_repos.output_ids

# 	print('batch', index)
# 	# print(out.shape)
# 	m1_p = out[tr, :].flatten()
# 	m1_l = labels[tr, :][:, tr].flatten()

# 	auc1s[index] = metrics.roc_auc_score(m1_l, m1_p)
# 	aupr1s[index] = metrics.average_precision_score(m1_l, m1_p)
# 	print('tr x tr', auc1s[index], aupr1s[index])

# 	m3_p = out[te, :].flatten()
# 	m3_l = labels[te, :][:, tr].flatten()

# 	auc3s[index] = metrics.roc_auc_score(m3_l, m3_p)
# 	aupr3s[index] = metrics.average_precision_score(m3_l, m3_p)
# 	print('te x tr', auc3s[index], aupr3s[index])

# test_auc, test_aupr = np.mean(auc3s), np.mean(aupr3s)
# print('Average')
# print('tr x tr', np.mean(auc1s), np.mean(aupr1s))
# print('te x tr', test_auc, test_aupr)