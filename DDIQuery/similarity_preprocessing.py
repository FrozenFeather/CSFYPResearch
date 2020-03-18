import scipy.io as io
import numpy as np
import math
from SNF import *

DATASET_TYPE = "DS"
DS1 = ("data/DS1/drug_drug_matrix.csv", \
	  ("data/DS1/chem_Jacarrd_sim.csv", "data/DS1/enzyme_Jacarrd_sim.csv",\
	   "data/DS1/offsideeffect_Jacarrd_sim.csv", "data/DS1/pathway_Jacarrd_sim.csv",\
	   "data/DS1/sideeffect_Jacarrd_sim.csv", "data/DS1/target_Jacarrd_sim.csv",\
	   "data/DS1/transporter_Jacarrd_sim.csv"))
DS2 = ("data/DS2/ddiMatrix.csv", ("data/DS2/simMatrix.csv"))
DS3_sim = (("data/DS3/ATCSimilarityMat.csv", "data/DS3/GOSimilarityMat.csv",\
		    "data/DS3/SideEffectSimilarityMat.csv", "data/DS3/chemicalSimilarityMat.csv",\
		    "data/DS3/distSimilarityMat.csv", "data/DS3/ligandSimilarityMat.csv",\
		    "data/DS3/seqSimilarityMat.csv"))
DS3_CRD = (("data/DS3/CRDInteractionMat.csv"), DS3_sim)
DS3_NCRD = (("data/DS3/NCRDInteractionMat.csv"), DS3_sim)

DATASET_FILENAMES = DS2

# Load the data file
def loadBMCData(filename):
	D = io.loadmat(filename)
	triple = D["DDI_triple"]
	binary = D["DDI_binary"]
	f_offside = D["offsides_feature"]
	pca_offside = D["pca_offisides"]
	f_structure = D["structure_feature"]
	pca_structure = D["pca_structure"]
	# return (binary, triple, f_offside, pca_offside, f_structure, pca_structure)
	return (binary, np.hstack((pca_offside, pca_structure)))

def loadDSData(filenames, include_header = False):
	ddi_filename, sim_filename = filenames
	try:
		interaction = np.loadtxt(ddi_filename,dtype=int,delimiter=",")
	except:
		interaction = np.loadtxt(ddi_filename,dtype=str,delimiter=",")[1:,1:].astype(np.uint8)

	if type(sim_filename) == str:
		try:
			drug_fea = np.loadtxt(sim_filename,dtype=float,delimiter=",")
		except:
			drug_fea = np.loadtxt(sim_filename,dtype=str,delimiter=",")[1:,1:].astype(np.single)
	else:
		K = 4
		t = 3
		ENTROPY_CUTOFF = 0.6
		drug_feas = {}
		entropies = []
		all_euclideanDist_Sim = {}
		for sim_i in sim_filename:
			feat_matrix = np.loadtxt(sim_i,dtype=str,delimiter=",").astype(np.single) + 0.001
			entropy, entropy_exclude_zero_sumRow, max_entropy = read_Sim_Calc_Entropy(sim_i, 1e-5)

			if entropy < ENTROPY_CUTOFF * math.log(len(feat_matrix), 2):
				for feat_type in drug_feas.keys():
					all_euclideanDist_Sim[feat_type + "," + sim_i] = euclidean_distance(drug_feas[feat_type], feat_matrix)
				drug_feas[sim_i] = feat_matrix
				entropies.append((sim_i, entropy))
		# drug_feas = [np.loadtxt(sim_i,dtype=str,delimiter=",").astype(np.single)+0.001 for sim_i in sim_filename]
		ranked_entropy_simType = [i[0] for i in sorted(entropies, key=lambda x:x[1])]
		final_drug_simType = removeRedundancy(ranked_entropy_simType,all_euclideanDist_Sim)
		drug_fea_mats = [drug_feas[i] for i in final_drug_simType]
		drug_fea = SNF(drug_fea_mats, K, t)
	# print(interaction.shape, drug_fea.shape)
	
	return (interaction, drug_fea)

def euclidean_distance(matA, matB):
	diff = np.array(matA)-np.array(matB)
	norm_sq = np.sum(diff ** 2)
	return 1/(1+math.sqrt(norm_sq))

def load_from_dataset():
	if DATASET_TYPE == 'DS':
		return loadDSData(DATASET_FILENAMES)
	else:
		return loadBMCData(DATASET_FILENAMES)

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
