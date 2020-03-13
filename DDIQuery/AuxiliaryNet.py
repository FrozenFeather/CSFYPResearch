################################################################
#
# drug2vector.py
# Main program
# Execute this file to train and evaluate the neural network
#
################################################################
import numpy as np
import scipy.io as io
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from batch import *
from similarity_preprocessing import *
from sklearn.linear_model import LinearRegression, Ridge
from drug2vector import Net
import sys

# Parameters
# Mode: 'whole': Train with the whole adjacent matrix(depreciated)
#		'normal': Train with the adjacent matrix with some '1's are eroded
#		'blind': Train with the submatrix of the adjacent matrix (CV enabled)
#		'blind_tuning': Same as 'blind' except run in different parameters
# epoch: Number of training epoch
# folds: Number of folds in cross-validation
# embed_size: Number of neurons in hidden layer(size of embedded vector)
# weight_decay: L2 Regularization Factor of the neural network
# ridge_alpha: L2 Regularization Factor of the linear extrapolation
# learning_rate: learning_rate of the neural network
# criterion: Loss function of the neural network

# weight_decay = 6.3e-5
folds = 10
embed_size = 100
ridge_alpha = 0.89
criterion = nn.BCELoss()

# Neural Network
# Contains 2 linear layer and with sigmoid activation layer
class AuxiliaryNet(nn.Module):
	def __init__(self, onehot_size, embed_size):
		super(AuxiliaryNet, self).__init__()
		self.fcA1 = nn.Linear(onehot_size, embed_size, bias=False)
		self.fcB1 = nn.Linear(embed_size, onehot_size * 2, bias=False)
		self.epoch = 180
		self.weight_decay = 1.2e-5
		self.learning_rate = 16.4e-3

	def forward(self, x):
		embed = self.fcA1(x)

		pred = self.fcB1(embed)
		s = nn.Sigmoid()
		sigmoid = s(pred)
		return sigmoid, pred, embed

	def getWeight(self):
		return self.fcA1.weight.data.numpy(), self.fcB1.weight.data.numpy()

	def setWeight(self, a, b):
		self.fcA1.weight.data = torch.FloatTensor(a)
		self.fcB1.weight.data = torch.FloatTensor(b)

def updateweight(target, train_feat, test_feat):
	reg = Ridge(alpha=ridge_alpha)
	reg.fit(train_feat, target)
	test_pred = reg.predict(test_feat)
	# print('lr score:', reg.score(train_feat, target))
	return test_pred

def load_auxiliary_set():
	batch = loadblinddata(1, folds)
	auc1s = np.zeros(folds)
	aupr1s = np.zeros(folds)
	auc3s = np.zeros(folds)
	aupr3s = np.zeros(folds)
	auc13s = np.zeros(folds)
	aupr13s = np.zeros(folds)
	
	for index, repos in enumerate(batch.repos):
		train_repos, test_repos = repos["train"], repos["test"]
		model = AuxiliaryNet(train_repos.onehot_size, embed_size)
		sim_matrix = load_sim_matrix(train_repos.input_ids)

		optimizer = optim.Adam(model.parameters(), weight_decay=model.weight_decay, lr=model.learning_rate)

		test_input, test_output_onehot = test_repos.miniBatch(568)

		for i in range(model.epoch):
			while train_repos.Epoch:
				train_input, train_output_onehot = train_repos.miniBatch(568)
				# print(train_input.shape)
				optimizer.zero_grad()
				sigmoid, pred, embed = model(train_input)

				ddi = train_output_onehot * 2
				sim = np.matmul(train_input, sim_matrix)
				output_label = torch.from_numpy(np.concatenate((ddi, sim)).astype(np.single))

				loss = criterion(sigmoid.flatten(), output_label.flatten())
				loss.backward()
				optimizer.step()

			train_repos.reset()

		
		# print(test_input.shape)
		# print(test_repos.input_feat.shape)

		# print(test_output_onehot.shape)
		# sigmoid, pred, embed = model(test_input)
		# out = sigmoid.data.numpy()
		# labels = test_output_onehot.data.numpy()
		# auc = metrics.roc_auc_score(labels, out)
		# aupr = metrics.average_precision_score(labels, out)
		# print(i, auc, aupr)


		tr = test_repos.input_ids
		te = test_repos.output_ids

		test_model = Net(test_repos.onehot_size, embed_size)
		u,v = model.getWeight()
		v = v[0:train_repos.onehot_size,:]
		new_u = updateweight(u.transpose(), test_repos.input_feat, test_repos.drug_feat).transpose()

		# print('size', len(tr)+len(te))
		# embed_vector = new_u.transpose()
		embed_vector = np.matmul(test_input.data.numpy(), new_u.transpose())
		m1 = np.matmul(embed_vector[tr, :], v.transpose())
		m3 = np.matmul(embed_vector[te, :], v.transpose())
		m13 = np.matmul(embed_vector, v.transpose())
		m2 = np.transpose(m3)
		reg = LinearRegression()
		reg.fit(embed_vector[tr, :], m2)
		new_v = np.zeros((test_repos.onehot_size, embed_size))
		new_v[tr, :] = v
		new_v[te, :] = reg.coef_

		test_model.setWeight(new_u, new_v)
		sigmoid, pred, embed = test_model(torch.FloatTensor(np.identity(test_repos.onehot_size)))
		out = sigmoid.data.numpy()
		# labels = test_output_onehot.data.numpy()
		labels = test_repos.adjcent_matrix

		print('batch', index)
		# print(out.shape)
		m1_p = out[tr, :][:, tr].flatten()
		m1_l = labels[tr, :][:, tr].flatten()

		auc1s[index] = metrics.roc_auc_score(m1_l, m1_p)
		aupr1s[index] = metrics.average_precision_score(m1_l, m1_p)
		print('tr x tr', auc1s[index], aupr1s[index])

		# m2_p = out[tr, :][:, te].flatten()
		# m2_l = labels[tr, :][:, te].flatten()

		# auc2 = metrics.roc_auc_score(m2_l, m2_p)
		# aupr2 = metrics.average_precision_score(m2_l, m2_p)
		# print('tr x te', auc2, aupr2)

		m3_p = out[te, :][:, tr].flatten()
		m3_l = labels[te, :][:, tr].flatten()

		auc3s[index] = metrics.roc_auc_score(m3_l, m3_p)
		aupr3s[index] = metrics.average_precision_score(m3_l, m3_p)
		print('te x tr', auc3s[index], aupr3s[index])

	test_auc, test_aupr = np.mean(auc3s), np.mean(aupr3s)
	print('Average')
	print('tr x tr', np.mean(auc1s), np.mean(aupr1s))
	print('te x tr', test_auc, test_aupr)

if __name__ == '__main__':
	load_auxiliary_set()
