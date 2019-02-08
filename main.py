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

def save_result(result, dir, cv_score):
	with open(dir, 'w') as f:
		f.write(str(cv_score) + '\n')
		for i in result:
			f.write(str(i[0]) + ' ' + str(i[1]) + '\n')

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
	
	G = s2v.read_graph(args)
	G = s2v.struc2vec.Graph(G, args.directed, args.workers)

	X,y,df = learning.load_df(args)
	cv_score = calc_cv_score(G, X,y,df, args)
	final_result(G, X,y,df, args, cv_score)



def main(args):
	s2v.execs2v(args)
	start(args)
	

    #region to debug==
    # weights_distances_r = ad.restoreVariableFromDisk('distances-r-'+str(1))
    # s2v.logging.info('{}'.format(weights_distances_r))
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = None)
    # G.calc_distances(compactDegree = args.OPT1)   
    # G.create_distances_network()
    # G.preprocess_parameters_random_walk()
    # G.simulate_walks(args.num_walks, args.walk_length)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = None)
    # G.simulate_walks(args.num_walks, args.walk_length)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = args.until_layer)
    # G.preprocess_neighbors_with_bfs_compact()
    # s2v.learn_embeddings(args)



def final_result(G, X,y,df, args, cv_score):
	c_proba = learning.set_classification(args, X, y, df)[0]
	final_proba = learning.sybil_rank(G, c_proba, args.pfinal)
	final_proba = normalizar_proba(final_proba)
	final_proba = sorted(final_proba.items(), key=operator.itemgetter(1), reverse=True)
	#print('Normalizado', final_proba) 
	save_result(final_proba, args.output_rank, cv_score)

def evaluate_score(y_expected, y_predict):
	max_v = 0
	max_t = 0
	logging.info("Evaluating fold...")
	for t in range(0, 80, 1):
		i = t*0.001
		y_predict_t = {k:(1 if v > i else 0) for k,v in y_predict.items() }
		#print(y_predict)
		y_predict_t = [y_predict_t[v] for v in y_expected.index]
		score = balanced_accuracy_score(y_expected, y_predict_t)
		if (score > max_v):
			max_v = score
			max_t = i
	logging.info("Fold evaluated: {}".format(max_v))
	return max_v 

def calc_cv_score(G, X,y,df, args):
	c_probas = learning.set_classification(args, X,y,df, folds=3)
	cv_score = []
	for c_proba in c_probas:
		final_proba = learning.sybil_rank(G, c_proba, args.pfinal)
		final_proba = normalizar_proba(final_proba)
		score = evaluate_score(y, final_proba)
		cv_score.append(score)
	
	return np.array(cv_score).mean()
	




def test_models(args):
	with ProcessPoolExecutor(max_workers=args.workers) as executor:
		outputs = ["steam_0", "steam_025", "steam_05", "steam_075", "steam_1"]
		pfinals = [0, 0.1, 0.25, 0.35, 0.5, 0.75, 0.9, 1]
		params_C = [0.01, 0.1, 1] 
		params_kernel = ['linear', 'poly', 'rbf'] 
		params_neigh = [1, 3, 7, 11] 
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



if __name__ == '__main__':
	args = parse_args()
	main(args)


