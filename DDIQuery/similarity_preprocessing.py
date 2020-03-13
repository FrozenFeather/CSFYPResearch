import scipy.io as io
import numpy as np
from SNF import SNF


def similarity_matrix(feat_matrix):
	n, f = feat_matrix.shape
	dist = np.zeros((n, n))
	max_dist = 0
	for i in range(n):
		for j in range(n):
			dist[i][j] = np.linalg.norm(feat_matrix[i,:] - feat_matrix[j,:])
			if dist[i][j] > max_dist:
				max_dist = dist[i][j]
	# return (max_dist-dist)/max_dist
	return 1/(1+dist)

def loadFeatures(filename, indices):
	D = io.loadmat(filename)
	return [similarity_matrix(D["pca_offisides"][indices]), similarity_matrix(D["pca_structure"][indices])]
	#  keys: "DDI_triple", "DDI_binary", "offsides_feature", "pca_offisides", 
	#		 "structure_feature", "pca_structure"

def gen_similarity_matrices(feat_matrices, K=2, t=1):
	return SNF([similarity_matrix(i) for i in feat_matrices], K, t)

def load_sim_matrix(indices):

	data_filename = 'data/DDI.mat'
	feat_matrices = loadFeatures(data_filename, indices)
	sim_matrix = SNF(feat_matrices, 2, 2)
	return sim_matrix
