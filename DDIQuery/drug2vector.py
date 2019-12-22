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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
from datetime import datetime
from batch import *
import sys
import matplotlib.pyplot as plt

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
mode = 'blind'
radius = 1
epoch = 80 if mode == 'blind_tuning' else 750
folds = 10
embed_size = 80
weight_decay = 6.3e-5
ridge_alpha = 0.89
learning_rate = 16.4e-3
criterion = nn.BCELoss()

# Neural Network
# Contains 2 linear layer and with sigmoid activation layer
class Net(nn.Module):
	def __init__(self, onehot_size, embed_size):
		super(Net, self).__init__()
		self.fcA1 = nn.Linear(onehot_size, embed_size, bias=False)

		self.fcB1 = nn.Linear(embed_size, onehot_size, bias=False)

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

# Depreciated evaluation method
def enrichment(order, labels):
	bins = np.zeros(shape=(order.shape[1]))
	count = order.shape[0]
	for i in range(count):
		row = list(order[i])
		l = labels[i]
		posi = row.index(l)
		bins[posi] += 1

	for i in range(1, bins.shape[0]):
		bins[i] += bins[i-1]
	return sum(bins) / bins.shape[0] / count

# Linear extrapolation of the unknown drugs based on the feature
def updateweight(target, train_feat, test_feat):
	reg = Ridge(alpha=ridge_alpha)
	reg.fit(train_feat, target)
	test_pred = reg.predict(test_feat)
	# print('lr score:', reg.score(train_feat, target))
	return test_pred

# Evaluation Method: Receiver-Operating Characteristic curve plotting
def export_roc_curve(y_true, y_pred, auc):
	fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
	plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
	plt.title('Receiver-Operating Curve')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.savefig('ROC.png')
	plt.clf()

# Evaluation Method: Precision-Recall curve plotting
def export_pr_curve(y_true, y_pred, aupr):
	prec, rec, threshold = metrics.precision_recall_curve(y_true, y_pred)
	plt.plot(rec, prec, label='PR curve (area = %0.3f)' % aupr)
	plt.title('Precision-Recall Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='lower right')
	plt.savefig('PRC.png')
	plt.clf()

# Train with the adjacent matrix with some '1's are eroded
def load_normal_set():

	train_repos, test_repos = loadnormaldata(radius)
	onehot_size = train_repos.onehot_size
	
	model = Net(onehot_size, embed_size)
	
	train_ids = test_repos.input_ids
	test_ids = test_repos.output_ids
	
	optimizer = optim.Adam(model.parameters())

	test_input, test_output_onehot = test_repos.miniBatch(568)
	i = 0
	for i in range(epoch):
		while train_repos.Epoch:
			train_input, train_output_onehot = train_repos.miniBatch(568)
			optimizer.zero_grad()
			sigmoid, pred, embed = model(train_input)
			loss = criterion(sigmoid.flatten(), train_output_onehot.flatten())
			loss.backward()
			optimizer.step()

		sigmoid, pred, embed = model(test_input)
		
		out = sigmoid.data.numpy().flatten()
		labels = test_output_onehot.data.numpy().flatten()
		auc = metrics.roc_auc_score(labels, out)
		aupr = metrics.average_precision_score(labels, out)
		print(i, auc, aupr)

		if i == epoch - 1:
			export_roc_curve(labels, out, auc)
			export_pr_curve(labels, out, aupr)
		
		train_repos.reset()

		whole_input = torch.FloatTensor(np.identity(onehot_size))
		sigmoid, pred, embed = model(whole_input)
		if i%10 == 9:
			np.save(str(i) + ".npy", embed.data.numpy())
			print("saved!")

		
#		'blind': Train with the submatrix of the adjacent matrix (CV enabled)
def load_blind_set():

	batch = loadblinddata(radius, folds)
	auc1s = np.zeros(folds)
	aupr1s = np.zeros(folds)
	auc3s = np.zeros(folds)
	aupr3s = np.zeros(folds)
	auc13s = np.zeros(folds)
	aupr13s = np.zeros(folds)
	
	for index, repos in enumerate(batch.repos):
		train_repos, test_repos = repos["train"], repos["test"]
		model = Net(train_repos.onehot_size, embed_size)

		optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)

		test_input, test_output_onehot = test_repos.miniBatch(568)

		for i in range(epoch):
			while train_repos.Epoch:
				train_input, train_output_onehot = train_repos.miniBatch(568)
				optimizer.zero_grad()
				sigmoid, pred, embed = model(train_input)
				loss = criterion(sigmoid.flatten(), train_output_onehot.flatten())
				loss.backward()
				optimizer.step()

			train_repos.reset()

		test_model = Net(test_repos.onehot_size, embed_size)
		u,v = model.getWeight()
		new_u = updateweight(u.transpose(), test_repos.input_feat, test_repos.drug_feat).transpose()

		tr = test_repos.input_ids
		te = test_repos.output_ids

		# print('size', len(tr)+len(te))
		# embed_vector = new_u.transpose()
		embed_vector = np.matmul(test_input.data.numpy(), new_u.transpose())
		m1 = np.matmul(embed_vector[tr, :], v.transpose())
		m3 = np.matmul(embed_vector[te, :], v.transpose())
		m13 = np.matmul(embed_vector, v.transpose())
		m2 = np.transpose(m3)
		# m12 = np.zeros((train_repos.onehot_size,test_repos.onehot_size))
		# m12[:, tr] = m1
		# m12[:, te] = m2
		# m12 = np.transpose(m13)
		# print(embed_vector[test_repos.input_ids].shape, m2.shape)
		# reg = Ridge(alpha=1e-3)
		reg = LinearRegression()
		reg.fit(embed_vector[tr, :], m2)
		# print(reg.score(embed_vector[tr, :], m2))
		# print(v.shape, reg.coef_.shape)
		new_v = np.zeros((test_repos.onehot_size, embed_size))
		# new_v = reg.coef_
		# m2 = np.matmul(embed_vector[tr,:], new_v[te,:].transpose())
		# m12 = np.matmul(embed_vector[tr,:], new_v.transpose())
		new_v[tr, :] = v
		new_v[te, :] = reg.coef_

		# m1_l = test_output_onehot.data.numpy()[tr,:][:,tr]
		# auc1 = metrics.roc_auc_score(m1_l.flatten(), m1.flatten())
		# aupr1 = metrics.average_precision_score(m1_l.flatten(), m1.flatten())
		# print('m1_t', auc1, aupr1)
		# m2_l = test_output_onehot.data.numpy()[tr,:][:,te]
		# auc2 = metrics.roc_auc_score(m2_l.flatten(), m2.flatten())
		# aupr2 = metrics.average_precision_score(m2_l.flatten(), m2.flatten())
		# print('m2_t', auc2, aupr2)
		# m3_l = test_output_onehot.data.numpy()[te,:][:,tr]
		# auc3 = metrics.roc_auc_score(m3_l.flatten(), m3.flatten())
		# aupr3 = metrics.average_precision_score(m3_l.flatten(), m3.flatten())
		# print('m3_t', auc3, aupr3)
		# m13_l = test_output_onehot.data.numpy()[:,tr]
		# auc13 = metrics.roc_auc_score(m13_l.flatten(), m13.flatten())
		# aupr13 = metrics.average_precision_score(m13_l.flatten(), m13.flatten())
		# print('m13_t', auc13, aupr13)
		# m12_l = test_output_onehot.data.numpy()[tr,:]
		# auc12 = metrics.roc_auc_score(m12_l.flatten(), m12.flatten())
		# aupr12 = metrics.average_precision_score(m12_l.flatten(), m12.flatten())
		# print('m12_t', auc12, aupr12)

		#new_v = np.linalg.pinv(updateweight(np.linalg.pinv(v).transpose(), test_repos.input_feat, test_repos.drug_feat)).transpose()
		test_model.setWeight(new_u, new_v)

		# sigmoid, pred, embed = test_model(test_input)
		sigmoid, pred, embed = test_model(torch.FloatTensor(np.identity(test_repos.onehot_size)))
		out = sigmoid.data.numpy()
		# labels = test_output_onehot.data.numpy()
		labels = test_repos.adjcent_matrix

		# tr = np.sort(test_repos.input_ids)
		# te = np.sort(test_repos.output_ids)
		
		print('batch', index, end=' ')
		m1_p = out[tr, :][:, tr].flatten()
		m1_l = labels[tr, :][:, tr].flatten()

		auc1s[index] = metrics.roc_auc_score(m1_l, m1_p)
		aupr1s[index] = metrics.average_precision_score(m1_l, m1_p)
		# print('tr x tr', auc1s[index], aupr1s[index])

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

		# m4_p = out[te, :][:, te].flatten()
		# m4_l = labels[te, :][:, te].flatten()

		# auc4 = metrics.roc_auc_score(m4_l, m4_p)
		# aupr4 = metrics.average_precision_score(m4_l, m4_p)
		# print('te x te', auc4, aupr4)
		
		# m13_p = out[:, tr].flatten()
		# m13_l = labels[:, tr].flatten()

		# auc13s[index] = metrics.roc_auc_score(m13_l, m13_p)
		# aupr13s[index] = metrics.average_precision_score(m13_l, m13_p)
		# print('All x tr', auc13s[index], aupr13s[index])

		# auc = metrics.roc_auc_score(labels.flatten(), out.flatten())
		# aupr = metrics.average_precision_score(labels.flatten(), out.flatten())
		# print('Overall', auc, aupr)
		# print(np.where(labels[tr, :][:, te]!=labels[te, :][:, tr].transpose()))

		# export_roc_curve(m13_l, m13_p, auc13)
		# export_pr_curve(m13_l, m13_p, aupr13)
			# export_roc_curve(labels, out, auc)
			# export_pr_curve(labels, out, aupr)
		
		# whole_input = torch.FloatTensor(np.identity(test_repos.onehot_size))
		# sigmoid, pred, embed = test_model(whole_input)
		# if i%10 == 9:
			# np.save(str(i) + ".npy", embed.data.numpy())
			# print("saved!")
	
	test_auc, test_aupr = np.mean(auc3s), np.mean(aupr3s)
	print('Average')
	# print('tr x tr', np.mean(auc1s), np.mean(aupr1s))
	print('te x tr', test_auc, test_aupr)
	# print('all x tr', np.mean(auc13s), np.mean(aupr13s))
	
	return test_auc, test_aupr
	
if __name__ == "__main__":
	np.random.seed()
	if mode == 'normal':
		load_normal_set()
	elif mode == 'blind':
		load_blind_set()
	elif mode == 'blind_tuning':
		aucs, auprs = [], []
		index = np.arange(0.86, 0.91, 0.01)
		for i in index:
			ridge_alpha = i
			print('ridge_alpha: ', ridge_alpha)
			auc, aupr = load_blind_set()
			aucs.append(auc)
			auprs.append(aupr)
			
		optimum_auc = np.argmax(aucs)
		optimum_aupr = np.argmax(auprs)
		plt.plot(index, np.asarray(aucs), label='AUC', color='red')
		plt.scatter(index[optimum_auc], aucs[optimum_auc], color='red')
		plt.annotate(aucs[optimum_auc], (index[optimum_auc], aucs[optimum_auc]))
		plt.plot(index, np.asarray(auprs), label='AUPR', color='blue')
		plt.scatter(index[optimum_aupr], auprs[optimum_aupr], color='blue')
		plt.annotate(auprs[optimum_aupr], (index[optimum_aupr], auprs[optimum_aupr]))
		plt.title('Test AUC & AUPR')
		plt.xlabel('weight decay')
		plt.ylabel('Area')
		plt.legend(loc='lower right')
		plt.savefig('TestAU.png')
		plt.clf()
			
			
