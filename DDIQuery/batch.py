################################################################
#
# batch.py
# Produce batch for training
#
################################################################
import scipy.io as io
import numpy as np
import torch 
import networkx as nx
from sklearn.model_selection import KFold
from similarity_preprocessing import *

# Repository: Batch management
class Repository():
	def __init__(self, adjcent_matrix, input_ids, output_ids, drug_onehot, drug_feat, output_type='ddi'):
		self.adjcent_matrix = adjcent_matrix
		self.drug_onehot = drug_onehot
		self.onehot_size = drug_onehot.shape[1]
		#those four attributes only used in blind test
		self.input_ids = input_ids
		self.output_ids = output_ids
		self.drug_feat = drug_feat
		self.input_feat = drug_feat[input_ids]
		self.output_feat = drug_feat[output_ids]

		self.output_type = output_type

		self.length = self.adjcent_matrix.shape[0]

		self.positions = np.arange(self.length)
		np.random.shuffle(self.positions)

		self.offset = 0
		self.Epoch = True

	# Generate batch 
	def miniBatch(self, batch_size):
		if self.offset + batch_size <= self.length:
			posi = self.positions[self.offset : self.offset + batch_size]
		else:
			posi = self.positions[self.offset : self.length]
			self.Epoch = False
			np.random.shuffle(self.positions)

		self.offset += batch_size

		feat_in = self.drug_onehot[posi]
		if self.output_type == 'ddi':
			feat_out = self.adjcent_matrix[posi]
		elif self.output_type == 'feat':
			feat_out = self.drug_feat[posi,:][:, posi]


		return torch.FloatTensor(feat_in), \
				torch.FloatTensor(feat_out)

	def reset(self):
		self.offset = 0
		self.Epoch = True

# BlindBatch: Cross validation batch management (For blind mode)
class BlindBatch():
	def __init__(self, adjcent_matrix, k, feature_matrix, radius, output_type = 'ddi'):
		self.adjcent_matrix = adjcent_matrix
		self.k = k
		self.feature_matrix = feature_matrix
		self.repos = []
		self.count = 0
		
		size = adjcent_matrix.shape[0]
		index = np.arange(size)
		kf = KFold(n_splits=self.k)
		for train_ids, test_ids in kf.split(index):
			m1, m2, m3, m4 = chooseblindtest(adjcent_matrix, train_ids, test_ids)
			
			train_matrix = dilute(m1, radius)
			
			train_onehot = np.identity(m1.shape[0])
			test_onehot = np.identity(adjcent_matrix.shape[0])

			train_repo = Repository(train_matrix, train_ids, train_ids, train_onehot, feature_matrix, output_type)
			test_repo = Repository(adjcent_matrix, train_ids, test_ids, test_onehot, feature_matrix, output_type)
			self.repos.append({'train': train_repo, 'test': test_repo})

class NDDBatch():
	def __init__(self, adjcent_matrix, sim_matrices):
		self.adjcent_matrix = adjcent_matrix
		self.similarities = sim_matrices
		self.onehot_size = sim_matrices.shape[1]
		self.current_batch = 0

		self.repos = []

		self.k = 10
		kf = KFold(n_splits=self.k)


		positives_r, positives_c = np.where(self.adjcent_matrix == 1)
		positives = np.array([x for x in zip(positives_r, positives_c)])
		negatives_r, negatives_c = np.where(self.adjcent_matrix == 0)
		negatives = np.array([x for x in zip(negatives_r, negatives_c)])

		positive_batch = kf.split(np.arange(positives.shape[0]))
		negative_batch = kf.split(np.arange(negatives.shape[0]))

		for i in range(self.k):
			train_positive_index, test_positive_index = next(positive_batch)
			train_negative_index, test_negative_index = next(negative_batch)
			train_repo = np.concatenate((positives[train_positive_index], negatives[train_negative_index]))
			test_repo = np.concatenate((positives[test_positive_index], negatives[test_negative_index]))
			np.random.shuffle(train_repo)
			np.random.shuffle(test_repo)

			self.repos.append({'train': train_repo, 'test': test_repo})

		self.positions = np.concatenate((positives, negatives))
		np.random.shuffle(self.positions)
		# 

		self.length = self.positions.shape[0]
		self.train_offset = 0
		self.test_offset = 0
		self.trainEpoch = True
		self.testEpoch = True


	# def miniBatch(self, batch_size):
	# 	if self.offset + batch_size <= self.length:
	# 		posi = self.positions[self.offset : self.offset + batch_size]
	# 	else:
	# 		posi = self.positions[self.offset : self.length]
	# 		self.Epoch = False

	# 	self.offset += batch_size

	# 	feat_in = []
	# 	feat_out = []
	# 	for row,col in posi:
	# 		drugA = self.similarities[row,:]
	# 		drugB = self.similarities[col,:]
	# 		joinfeat = np.concatenate((drugA,drugB))
	# 		feat_in.append(joinfeat)
	# 		feat_out.append(self.adjcent_matrix[row, col])
	# 	feat_out = np.array(feat_out)

	# 	return torch.FloatTensor(feat_in), torch.FloatTensor(feat_out)

	def trainBatch(self, batch_size):
		positions = self.repos[self.current_batch]['train']
		if self.train_offset + batch_size <= positions.shape[0]:
			posi = positions[self.train_offset : self.train_offset + batch_size]
		else:
			posi = positions[self.train_offset : positions.shape[0]]
			self.trainEpoch = False
		self.train_offset += batch_size

		feat_in = []
		feat_out = []
		for row, col in posi:
			drugA = self.similarities[row,:]
			drugB = self.similarities[col,:]
			joinfeat = np.concatenate((drugA,drugB))
			feat_in.append(joinfeat)
			feat_out.append(self.adjcent_matrix[row, col])
		feat_out = np.array(feat_out)

		return torch.FloatTensor(feat_in), torch.FloatTensor(feat_out)

	def testBatch(self, batch_size):
		positions = self.repos[self.current_batch]['test']
		if self.test_offset + batch_size <= positions.shape[0]:
			posi = positions[self.test_offset : self.test_offset + batch_size]
		else:
			posi = positions[self.test_offset : positions.shape[0]]
			self.testEpoch = False
		self.test_offset += batch_size

		feat_in = []
		feat_out = []
		for row, col in posi:
			drugA = self.similarities[row,:]
			drugB = self.similarities[col,:]
			joinfeat = np.concatenate((drugA,drugB))
			feat_in.append(joinfeat)
			feat_out.append(self.adjcent_matrix[row, col])
		feat_out = np.array(feat_out)

		return torch.FloatTensor(feat_in), torch.FloatTensor(feat_out)

	def next_batch(self):
		self.current_batch += 1
		if self.current_batch >= self.k:
			self.current_batch = 0

	def reset_train(self):
		self.train_offset = 0
		self.trainEpoch = True

	def reset_test(self):
		self.test_offset = 0
		self.testEpoch = True
	


# same as np.flatten
def flat(matrix):
	row, col = np.where(matrix>0)
	m  = matrix
	for i in range(len(row)):
		m[row[i], col[i]] = 1
	return m

# Depreciated
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

# Erode '1's in the adjacent matrix(for normal mode)
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

def chooseNDDtest(adjcent_matrix, ratio):
	length = adjcent_matrix.shape[0]
	test_length = int(length * ratio)
	ids = list(range(length))
	np.random.shuffle(ids)
	test_ids = ids[:test_length]
	train_ids = ids[test_length:]
	
	m1 = adjcent_matrix[train_ids, :][:, train_ids]
	m2 = adjcent_matrix[train_ids, :][:, test_ids]
	m3 = adjcent_matrix[test_ids, :][:, train_ids]
	m4 = adjcent_matrix[test_ids, :][:, test_ids]
	return m1, m2, m3, m4, train_ids, test_ids

def loadnormaldata(radius):
	adjcent_matrix, _, f1, f2, f3, f4 = loadBMCData("data/DDI.mat")
	# adjcent_matrix, f3 = loadBMCData("data/DDI.mat")

	test_posi = choosenormaltest(adjcent_matrix, 0.15)

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
	adjcent_matrix, feature_matrix = load_from_dataset()
	batch = BlindBatch(adjcent_matrix, no_of_splits, feature_matrix, radius)
	return batch

def loadNDDdata():
	adjcent_matrix, sim_matrix = loadBMCData("data/DDI.mat")
	# sim_matrix = similarity_matrix(np.hstack((f2,f4)))
	m1, m2, m3, m4, train_ids, test_ids = chooseNDDtest(adjcent_matrix, 0.1)
	
	onehot = np.identity(adjcent_matrix.shape[0])

	return NDDBatch(m1, sim_matrix), \
			NDDBatch(adjcent_matrix, sim_matrix)

if __name__ == "__main__":
	train, test = loadblinddata(1)
	test.miniBatch(5000)
