import scipy.io as io
import numpy as np
import torch 
import networkx as nx
from sklearn.model_selection import KFold

test_ratio = 0.15

class Repository():
	def __init__(self, adjcent_matrix, input_ids, output_ids, drug_onehot, drug_feat):
		self.adjcent_matrix = adjcent_matrix
		self.drug_onehot = drug_onehot
		self.onehot_size = drug_onehot.shape[1]
		#those four attributes only used in blind test
		self.input_ids = input_ids
		self.output_ids = output_ids
		self.drug_feat = drug_feat
		self.input_feat = drug_feat[input_ids]
		self.output_feat = drug_feat[output_ids]

		self.length = self.adjcent_matrix.shape[0]

		self.positions = np.arange(self.length)
		np.random.shuffle(self.positions)

		self.offset = 0
		self.Epoch = True


	def miniBatch(self, batch_size):
		if self.offset + batch_size <= self.length:
			posi = self.positions[self.offset : self.offset + batch_size]
		else:
			posi = self.positions[self.offset : self.length]
			self.Epoch = False
			np.random.shuffle(self.positions)

		self.offset += batch_size

		feat_in = self.drug_onehot[posi]
		feat_out = self.adjcent_matrix[posi]


		return torch.FloatTensor(feat_in), \
				torch.FloatTensor(feat_out)

	def reset(self):
		self.offset = 0
		self.Epoch = True

class BlindBatch():
	def __init__(self, adjcent_matrix, k, feature_matrix, radius):
		self.adjcent_matrix = adjcent_matrix
		self.k = k
		self.feature_matrix = feature_matrix
		self.repos = []
		self.count = 0
		
		size = adjcent_matrix.shape[0]
		index = np.arange(size)
		kf = KFold(n_splits=k)
		for train_ids, test_ids in kf.split(index):
			m1, m2, m3, m4 = chooseblindtest(adjcent_matrix, train_ids, test_ids)
			
			train_matrix = dilute(m1, radius)
			
			train_onehot = np.identity(m1.shape[0])
			test_onehot = np.identity(adjcent_matrix.shape[0])

			train_repo = Repository(train_matrix, train_ids, train_ids, train_onehot, feature_matrix)
			test_repo = Repository(adjcent_matrix, train_ids, test_ids, test_onehot, feature_matrix)
			self.repos.append({'train': train_repo, 'test': test_repo})
	

def loadBMCData(filename):
	D = io.loadmat(filename)
	triple = D["DDI_triple"]
	binary = D["DDI_binary"]
	f_offside = D["offsides_feature"]
	pca_offside = D["pca_offisides"]
	f_structure = D["structure_feature"]
	pca_structure = D["pca_structure"]
	return binary, triple, f_offside, pca_offside, f_structure, pca_structure

def flat(matrix):
	row, col = np.where(matrix>0)
	m  = matrix
	for i in range(len(row)):
		m[row[i], col[i]] = 1
	return m

def dilute(adjcent_matrix, step):
	m = adjcent_matrix + np.identity(adjcent_matrix.shape[0])
	temp = adjcent_matrix + np.identity(adjcent_matrix.shape[0])
	m_list = [m]
	for i in range(1, step):
		m  = np.matmul(m, temp)
		m_list.append(m)
	
	temp = m_list[0]
	for i in range(1, step):
		temp += m_list[i]
	return flat(temp) - np.identity(adjcent_matrix.shape[0])


def erosion(adjcent_matrix, positions):
	m  = np.copy(adjcent_matrix)
	for row, col in positions:
		m[row, col] = 0
	return m

def choosenormaltest(adjcent_matrix, ratio):
	rows, cols = np.where(adjcent_matrix == 1)
	positions = np.array([x for x in zip(rows, cols)])
	np.random.shuffle(positions)
	length = len(positions)
	test_length = int(length * ratio)
	test_posi = positions[:test_length]
	return test_posi

def chooseblindtest(adjcent_matrix, train_ids, test_ids):
	m1 = adjcent_matrix[train_ids, :][:, train_ids]
	m2 = adjcent_matrix[train_ids, :][:, test_ids]
	m3 = adjcent_matrix[test_ids, :][:, train_ids]
	m4 = adjcent_matrix[test_ids, :][:, test_ids]
	return m1, m2, m3, m4

def loadnormaldata(radius):
	adjcent_matrix, _, f1, f2, f3, f4 = loadBMCData("data/DDI.mat")
	# adjcent_matrix, f3 = loadBMCData("data/DDI.mat")

	test_posi = choosenormaltest(adjcent_matrix, test_ratio)

	test_matrix = np.zeros(shape=adjcent_matrix.shape)
	for row, col in test_posi:
		test_matrix[row, col] = 1

	range_matrix = dilute(adjcent_matrix, radius)
	train_matrix = erosion(range_matrix, test_posi)
	
	onehot = np.identity(adjcent_matrix.shape[0])
	feat = f3

	train_in = np.where(train_matrix.sum(axis=1) > 0)
	train_out = np.where(train_matrix.sum(axis=0) > 0)

	test_in = np.where(test_matrix.sum(axis=1) > 0)
	test_out = np.where(test_matrix.sum(axis=0) > 0)	

	print("original number of 1: " + str(np.where(train_matrix == 1)[0].size))
	print("test number of 1: " + str(np.where(adjcent_matrix == 1)[0].size))
	
	return Repository(train_matrix, train_in, train_out, onehot, feat), \
			Repository(adjcent_matrix, test_in, test_out, onehot, feat)

def loadwholedata(radius):
	adjcent_matrix, _, f1, f2, f3, f4 = loadBMCData("data/DDI.mat")
	
	test_posi = choosenormaltest(adjcent_matrix, 0.1)

	test_matrix = np.zeros(shape=adjcent_matrix.shape)
	for row, col in test_posi:
		test_matrix[row, col] = 1

	train_matrix = dilute(adjcent_matrix, radius)
	
	onehot = np.identity(adjcent_matrix.shape[0])
	feat = f3

	train_in = np.where(train_matrix.sum(axis=1) > 0)
	train_out = np.where(train_matrix.sum(axis=0) > 0)

	test_in = np.where(test_matrix.sum(axis=1) > 0)
	test_out = np.where(test_matrix.sum(axis=0) > 0)	
	
	return Repository(train_matrix, train_in, train_out, onehot, feat), \
			Repository(adjcent_matrix, test_in, test_out, onehot, feat)

def loadblinddata(radius, no_of_splits):
	adjcent_matrix, _, f1, f2, f3, f4 = loadBMCData("data/DDI.mat")
	feature_matrix = np.hstack((f2,f4))
	batch = BlindBatch(adjcent_matrix, no_of_splits, feature_matrix, radius)
	return batch

if __name__ == "__main__":
	train, test = loadblinddata(1)
	test.miniBatch(5000)
