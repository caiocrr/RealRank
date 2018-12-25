# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 00:45:31 2018

@author: Caio Ramos
"""

import sys
sys.path.append("struc2vec/src")
import main as s2v
class args: 
    input = "data/steam.edgelist";
    output = "output/steam.struc2vec";
    dimensions = 128;
    walk_length = 80;
    num_walks = 10;
    window_size = 10;
    until_layer = None;
    iter = 5;
    workers = 4;
    weighted = False;
    directed = False;
    OPT1 = True;
    OPT2 = True;
    OPT3 = True;
    
s2v.execs2v(args)
