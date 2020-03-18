import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

from drug2vector import Net, updateweight
from batch import Repository, NDDBatch, chooseNDDtest, loadblinddata
from NDD import NDD
from similarity_preprocessing import load_from_dataset

d2v_epoch = 250
ndd_epoch = 200
folds = 10
embed_size = 120
weight_decay = 8e-5
# weight_decay = 3.5e-4
ridge_alpha = 1.2
learning_rate = 1.2e-2
criterion = nn.BCELoss()


def load_d2v():
	interaction, feature = load_from_dataset()
	onehot_size = feature.shape[0]
	repo = Repository(interaction, np.arange(onehot_size), np.arange(onehot_size), np.identity(onehot_size), feature, output_type = 'feat')

	d2v = Net(onehot_size, embed_size)
	d2v_optimizer = optim.Adam(d2v.parameters(), weight_decay=weight_decay, lr=learning_rate)

	for i in range(d2v_epoch):
		while repo.Epoch:
			train_input, train_output_onehot = repo.miniBatch(1000)
			d2v_optimizer.zero_grad()
			sigmoid, pred, embed = d2v(train_input)
			loss = criterion(sigmoid.flatten(), train_output_onehot.flatten() * 3)
			loss.backward()
			d2v_optimizer.step()

		repo.reset()

	_, _, embed_matrix = d2v(torch.FloatTensor(np.identity(onehot_size)))

	ndd_model = NDD(embed_size)

	ndd_batch = NDDBatch(interaction, torch.FloatTensor(embed_matrix.data.numpy()))
	ndd_batch.next_batch()

	ndd_optimizer = optim.Adam(ndd_model.parameters(), weight_decay=weight_decay, lr=learning_rate)

	for i in range(ndd_epoch):
		if not ndd_batch.trainEpoch:
			ndd_batch.reset_train()

		train_input, train_output_onehot = ndd_batch.trainBatch(500)
		ndd_optimizer.zero_grad()
		pred = ndd_model(train_input)
		loss = criterion(pred.flatten(), train_output_onehot.flatten())
		loss.backward()
		ndd_optimizer.step()
		# ndd_batch.reset()

		if not ndd_batch.testEpoch:
			ndd_batch.reset_test()
		test_input, test_output_onehot = ndd_batch.testBatch(200)
		pred = ndd_model(test_input)
		out = pred.data.numpy()
		labels = test_output_onehot.data.numpy()
		# print(labels)

		auc = metrics.roc_auc_score(labels.flatten(), out.flatten())
		aupr = metrics.average_precision_score(labels.flatten(), out.flatten())
		print('Overall', auc, aupr)

def load_blind_d2v():

	batch = loadblinddata(1, folds)
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

		test_input, test_output_onehot = test_repos.miniBatch(1000)

		for i in range(d2v_epoch):
			while train_repos.Epoch:
				train_input, train_output_onehot = train_repos.miniBatch(1000)
				optimizer.zero_grad()
				sigmoid, pred, embed = model(train_input)
				loss = criterion(sigmoid.flatten(), train_output_onehot.flatten() * 3)
				loss.backward()
				optimizer.step()

			train_repos.reset()

		u,v = model.getWeight()
		embed_matrix = updateweight(u.transpose(), test_repos.input_feat, test_repos.drug_feat)

		tr = test_repos.input_ids
		te = test_repos.output_ids

		# print('size', len(tr)+len(te))
		# embed_vector = new_u.transpose()

		ndd_model = NDD(embed_size)

		ndd_batch = NDDBatch(train_repos.adjcent_matrix, torch.FloatTensor(embed_matrix))
		ndd_batch.next_batch()

		ndd_optimizer = optim.Adam(ndd_model.parameters(), weight_decay=weight_decay, lr=learning_rate)

		for i in range(ndd_epoch):
			if not ndd_batch.trainEpoch:
				ndd_batch.reset_train()

			train_input, train_output_onehot = ndd_batch.trainBatch(500)
			ndd_optimizer.zero_grad()
			pred = ndd_model(train_input)
			loss = criterion(pred.flatten(), train_output_onehot.flatten())
			loss.backward()
			ndd_optimizer.step()
			if not ndd_batch.testEpoch:
				ndd_batch.reset_test()
			test_input, test_output_onehot = ndd_batch.testBatch(200)
			pred = ndd_model(test_input)
			out = pred.data.numpy()
			labels = test_output_onehot.data.numpy()
			# print(labels)

			auc = metrics.roc_auc_score(labels.flatten(), out.flatten())
			aupr = metrics.average_precision_score(labels.flatten(), out.flatten())
			print('Overall', auc, aupr)



if __name__ == "__main__":
	np.random.seed()
	embed_matrix = load_blind_d2v()