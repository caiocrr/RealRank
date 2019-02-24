# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 00:45:31 2018

@author: Caio Ramos
"""

import sys
sys.path.append("struc2vec/src")
import s2v
import learning
import argparse, logging
import algorithms_distances as ad
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import operator
import copy
from sklearn.metrics import f1_score, balanced_accuracy_score

logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def parse_args():
	'''
	Parses the struc2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run struc2vec.")

	parser.add_argument('--input', nargs='?', default='social_networks/common_friends/social_network.edgelist',
	                    help='Input graph path')
	
	parser.add_argument('--train', nargs='?', default='social_networks/common_friends/social_network.train',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='social_networks/common_friends/social_network.struc2vec',
	                    help='Embeddings path')

	parser.add_argument('--output-rank', nargs='?', default='social_networks/common_friends/social_network.realrank',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 5.')

	parser.add_argument('--until-layer', type=int, default=2,
                    	help='Calculation until the layer. Default is 2')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD. Default is 5')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 4.')

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	
	parser.add_argument('--pcommonf', type=float, default=0.5, 
						help='Weight to set initial state of Sybil Rank: 0 for 100% degree similarity and 1 for 100% common friends similarity. Default is 0.5.')
	
	parser.add_argument('--pfinal', type=float, default=0.5, 
						help='Weight to set the importance of the Logistic Regression Classification for the final result. P=1 for 100% logistic regression.')

	parser.add_argument('--model', nargs='?', default='lr', 
						help='Classification model. Options: Logistic Regression (lr), K-Nearest-Neighbors (knn) and SVM (svm). Default is LR.')
	
	parser.add_argument('--nneigh', type=int, default=5, 
						help='Number of Nearest Neighbors of KNN model. Default is 5.')

	parser.add_argument('--kernel', nargs='?', default='rbf', 
						help='Kernel to use on SVM Classifier. Options: Linear(lin), Polynomial (poly), Radius Basis Function (rbf). Default is RBF.')

	parser.add_argument('--C', type=float, default=1.0,
						help='Penalty parameter of the error term of SVM Classifier')

	parser.add_argument('--calculated-distances', type=bool, default=False,
						help='Set True to utilize the previous calculated similarity between vertices. This is the biggest part of execution.')

	parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1. Default is True')
	parser.add_argument('--OPT2', default=True, type=bool,
                      help='optimization 2. Default is True')
	parser.add_argument('--OPT3', default=True, type=bool,
                      help='optimization 3. Default is True')	 
	return parser.parse_args()

def save_result(result, dir, cv_ba, cv_f1):
	logging.info("Salvando {} scores: ba: {} f1: {}".format(dir, cv_ba, cv_f1))
	with open(dir, 'w') as f:
		f.write(str(cv_ba) + "," + str(cv_f1) + '\n')
		for i in result:
			f.write(str(i[0]) + ' ' + str(i[1]) + '\n')

def save_cv_result(dir, cv_ba, cv_f1):
	logging.info("Salvando {} scores: ba: {} f1: {}".format(dir, cv_ba, cv_f1))
	with open(dir+".cv", 'w') as f:
		f.write(str(cv_ba)+ "," + str(cv_f1))

def convertToVertexDict(prob):
	return {k:v for k,v in enumerate(prob)}

def normalizar_proba(prob):
	v_a = [v for k,v in prob.items()]
	max_v = max(v_a)
	min_v = min(v_a)
	if((max_v - min_v) > 0):
		return {k: (v-min_v) / (max_v - min_v) for k,v in prob.items()}
	return prob
	

def start(args):
	logging.info("Classificando " + args.output_rank)
	
	try:
		G = s2v.read_graph(args)
		G = s2v.struc2vec.Graph(G, args.directed, args.workers)

		X,y,df = learning.load_df(args)
		cv_ba, cv_f1 = calc_cv_score(G, X,y,df, args)
		final_result(G, X,y,df, args, cv_ba, cv_f1)
		#save_cv_result(args.output_rank, cv_ba, cv_f1)
	except:
		logging.info('Erro: {}'.format(sys.exc_info()))
	

def start_cv(args):
	logging.info("Classificando " + args.output_rank)
	
	try:
		G = s2v.read_graph(args)
		G = s2v.struc2vec.Graph(G, args.directed, args.workers)

		X,y,df = learning.load_df(args)
		cv_score = calc_cv_score_compare(G, X,y,df, args)
		print(cv_score)
	except:
		logging.info('Erro: {}'.format(sys.exc_info()))

def main(args):
	s2v.execs2v(args)
	start(args)
	

    #region to debug==
    # weights_distances_r = ad.restoreVariableFromDisk('distances-r-'+str(1))
    # s2v.logging.info('{}'.format(weights_distances_r))
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = None)
    # G.calc_distances(compactDegree = args.OPT1)   
    # G.create_distances_network()90
    # G.preprocess_parameters_random_walk()
    # G.simulate_walks(args.num_walks, args.walk_length)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = None)
    # G.simulate_walks(args.num_walks, args.walk_length)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = args.until_layer)
    # G.preprocess_neighbors_with_bfs_compact()
    # s2v.learn_embeddings(args)



def final_result(G, X,y,df, args, cv_ba, cv_f1):
	c_proba = learning.set_classification(args, X, y, df)[0]['result']
	final_proba = learning.sybil_rank(G, c_proba, args.pfinal)
	final_proba = normalizar_proba(final_proba)
	final_proba = sorted(final_proba.items(), key=operator.itemgetter(1), reverse=True)
	#print('Normalizado', final_proba) 
	save_result(final_proba, args.output_rank, cv_ba, cv_f1)

def evaluate_score(y_expected, y_predict, test_index):
	max_ba = 0
	max_f1 = 0
	max_t_ba = None
	max_t_f1 = None

	y_expected = y_expected.loc[test_index].sort_index().values
	y_predict = sorted({k:v for k,v in y_predict.items() if k in test_index}.items(), key=operator.itemgetter(0))
	logging.info("Evaluating fold...")
	for i in np.arange(0, 0.5, 0.001):
		y_predict_t = [(1 if v > i else 0) for k,v in y_predict ]
		ba = balanced_accuracy_score(np.logical_not(y_expected).astype(int), np.logical_not(y_predict_t).astype(int))
		f1 = f1_score(np.logical_not(y_expected).astype(int), np.logical_not(y_predict_t).astype(int))
		if (ba > max_ba):
			max_ba = ba
			max_t_ba = i
			
		if (f1 > max_f1):
			max_f1 = f1
			max_t_f1 = i
	logging.info("Fold evaluated: ba: {} / f1: {}".format(max_ba, max_f1))
	return max_ba, max_f1

def calc_cv_score(G, X,y,df, args):
	c_probas_with_test_index = learning.set_classification(args, X,y,df, folds=10)
	cv_ba = []
	cv_f1 = []
	for c_proba_with_test_index in c_probas_with_test_index:
		c_proba = c_proba_with_test_index['result']
		test_index = c_proba_with_test_index['test_index']
		final_proba = learning.sybil_rank(G, c_proba, args.pfinal)
		final_proba = normalizar_proba(final_proba)
		ba,f1 = evaluate_score(y, final_proba, test_index)
		cv_ba.append(ba)
		cv_f1.append(f1)
	
	return np.array(cv_ba).mean(), np.array(cv_f1).mean()



def calc_cv_score_compare(G, X,y,df, args):
	c_probas_with_test_index_01 = learning.set_classification_folds(args, X,y,df, 0.1)
	c_probas_with_test_index_02 = learning.set_classification_folds(args, X,y,df, 0.2)
	c_probas_with_test_index_03 = learning.set_classification_folds(args, X,y,df, 0.3)
	c_probas_with_test_index_05 = learning.set_classification_folds(args, X,y,df, 0.5)
	c_probas_with_test_index_07 = learning.set_classification_folds(args, X,y,df, 0.7)
	c_probas_with_test_index_09 = learning.set_classification_folds(args, X,y,df, 0.9)

	c_probas_with_test_index_array = [c_probas_with_test_index_01,c_probas_with_test_index_02,c_probas_with_test_index_03,c_probas_with_test_index_05,c_probas_with_test_index_07,c_probas_with_test_index_09]

	
	cv_results = []
	labels = ['01','02','03','05','07','09']
	for c_probas_with_test_index, label in zip(c_probas_with_test_index_array, labels):
		cv_ba = []
		cv_f1 = []
		for c_proba_with_test_index in c_probas_with_test_index:
			c_proba = c_proba_with_test_index['result']
			test_index = c_proba_with_test_index['test_index']
			final_proba = learning.sybil_rank(G, c_proba, args.pfinal)
			final_proba = normalizar_proba(final_proba)
			ba,f1 = evaluate_score(y, final_proba, test_index)
			cv_ba.append(ba)
			cv_f1.append(f1)
	
		cv_results.append((label, np.array(cv_ba).mean(), np.array(cv_f1).mean()))
	return cv_results


def test_models(args):
	with ProcessPoolExecutor(max_workers=args.workers) as executor:
		outputs = ["steam_0", "steam_025", "steam_05", "steam_075", "steam_1"]
		pfinals = [0, 0.25, 0.5, 0.75, 1]
		params_C = [0.01, 0.1, 1] 
		params_kernel = ['linear', 'poly', 'rbf'] 
		params_neigh = [1, 3, 7] 
		for output in outputs:
			args.output = "data/"+output+".struc2vec"
			for pfinal in pfinals:
				args.pfinal = pfinal
				args.model = "lr"

				for c in params_C:
					args.C = c
					args.output_rank = "data/"+output+"_"+str(args.pfinal)+"_model_"+args.model+"_C_"+str(args.C)+".realrank"
					lr_args = copy.deepcopy(args)
					executor.submit(start,lr_args)
					#start(lr_args)

				args.model = "svm"
				for c in params_C:
					args.C = c
					for kernel in params_kernel:
						args.kernel = kernel
						args.output_rank = "data/"+output+"_"+str(args.pfinal)+"_model_"+args.model+"_C_"+str(args.C)+ "_kernel_"+str(args.kernel)+".realrank"
						svm_args =  copy.deepcopy(args)
						executor.submit(start,svm_args)
						#start(args)
						
				args.model = "knn"
				for neigh in params_neigh:
					args.nneigh = neigh
					args.output_rank = "data/"+output+"_"+str(args.pfinal)+"_model_"+args.model+"_nneigh_"+str(args.nneigh)+".realrank"
					knn_args =  copy.deepcopy(args)
					executor.submit(start,knn_args)
					#start(args)


def test_cv(args):
	#steam_0_0.25_model_svm_C_1_kernel_poly.realrank
	args.pfinal = 0.75
	args.output = "data/steam_075.struc2vec"
	#args.C = 0.01
	args.model = "knn"
	args.nneigh = 3
	args.output_rank = "data/cv/steam_1_"+str(args.pfinal)+"_model_"+args.model+"_C_"+str(args.C)+".realrank"
	svm_args =  copy.deepcopy(args)
	start_cv(svm_args)
	#start(args)


if __name__ == '__main__':
	args = parse_args()
	main(args)
	#test_models(args)
	#test_cv(args)

