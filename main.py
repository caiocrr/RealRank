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

def parse_args():
	'''
	Parses the struc2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run struc2vec.")

	parser.add_argument('--input', nargs='?', default='social_networks/social_network.edgelist',
	                    help='Input graph path', required=True)
	
	parser.add_argument('--train', nargs='?', default='social_networks/social_network.train',
	                    help='Input graph path', required=True)

	parser.add_argument('--output', nargs='?', default='social_networks/social_network.struc2vec',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=32,
	                    help='Number of dimensions. Default is 32.')

	parser.add_argument('--walk-length', type=int, default=20,
	                    help='Length of walk per source. Default is 20.')

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

	parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1. Default is True')
	parser.add_argument('--OPT2', default=True, type=bool,
                      help='optimization 2. Default is True')
	parser.add_argument('--OPT3', default=True, type=bool,
                      help='optimization 3. Default is True')	 
	return parser.parse_args()

def main(args):
	s2v.execs2v(args)
	learning.set_classification(args.output, args.train)
	learning.sybil_rank()



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

if __name__ == '__main__':
	args = parse_args()
	main(args)
