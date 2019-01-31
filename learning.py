import pandas as pd
import numpy as np
import json
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from math import log, floor

def set_classification(input, train, args):
        df = pd.read_csv(input, skiprows=1, sep=' ', header=None, index_col=0).sort_index()
        df_train = pd.read_csv(train, sep=' ', header=None, index_col=0).sort_index()
        df_train.columns = ['confiavel']
        df_train = pd.concat([df, df_train], axis=1, join='inner')
        y = df_train.loc[:, 'confiavel'].values
        X = df_train.drop(['confiavel'], axis=1)
        result = None
        if (args.model == "knn"):
                nneigh = args.nneigh
                knn = KNN(n_neighbors=nneigh).fit(X,y)
                result = pd.DataFrame(knn.predict(df).T, index=df.index).to_dict()[0]
        elif(args.model == "svm"):
                C = args.C
                kernel = args.kernel
                svm = SVC(C=C, random_state=0, kernel=kernel, class_weight='balanced').fit(X,y)
                result = pd.DataFrame(svm.predict(df).T, index=df.index).to_dict()[0]
        else:
                lr = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced').fit(X,y)
                result = pd.DataFrame(lr.predict_proba(df).T[1], index=df.index).to_dict()[0]
        
        return result

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
