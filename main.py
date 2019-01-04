# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 00:45:31 2018

@author: Caio Ramos
"""

import sys
sys.path.append("struc2vec/src")
import s2v
class args: 
    input = "data/steam.edgelist";
    output = "output/steam.struc2vec";
    dimensions = 128;
    walk_length = 80;
    num_walks = 10;
    window_size = 10; #importante
    until_layer = 2; #importante
    iter = 5;
    workers = 4;
    weighted = False;
    directed = False;
    OPT1 = True; #importante
    OPT2 = True; #importante
    OPT3 = True; #importante
    pcommonf = 0.5;
    
    

def main():
    s2v.execs2v(args)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = None)
    # G.simulate_walks(args.num_walks, args.walk_length)
    # G = s2v.read_graph(args)
    # G = s2v.struc2vec.Graph(G, args.directed, args.workers, untilLayer = args.until_layer)
    # G.preprocess_neighbors_with_bfs_compact()

    #s2v.learn_embeddings(args)

if __name__ == '__main__':
    main()
