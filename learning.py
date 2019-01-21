import pandas as pd
import numpy as np
import json
import operator
from sklearn.linear_model import LogisticRegression
from math import log, floor

def set_classification(input, train):
        df = pd.read_csv(input, skiprows=1, sep=' ', header=None, index_col=0).sort_index()
        df_train = pd.read_csv(train, sep=' ', header=None, index_col=0).sort_index()
        df_train.columns = ['confiavel']
        df_train = pd.concat([df, df_train], axis=1, join='inner')
        X = df_train.loc[:, [1,2]].values
        y = df_train.loc[:, 'confiavel'].values
        lr = LogisticRegression(random_state=0, solver='lbfgs').fit(X,y)
        return lr.predict_proba(df)

def sybil_rank(G, init_p, p_lr_syb):
        G = G.G
        n_iter = int(log(len(G.keys())))
        i_base = init_p
        for i in range(n_iter):
                i_atual = {}
                for v in G.keys():
                        #somar os pesos dos vizinhos divididos por cada grau
                        c_vizinhos = [i_base[j]/len(G[j]) for j in G[v]]
                        i_atual[v] = p_lr_syb*i_base[v] + (1-p_lr_syb)*np.sum(c_vizinhos)
                i_base = i_atual
        return i_base
