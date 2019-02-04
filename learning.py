import pandas as pd
import numpy as np
import json
import operator
import math,logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from math import log, floor
from sklearn.model_selection import cross_val_score

def set_classification(args):
        input = args.output #resultado (output) do s2v
        logging.info('Iniciando classificacao da rede')
        train = args.train
        df = pd.read_csv(input, skiprows=1, sep=' ', header=None, index_col=0).sort_index()
        df_train = pd.read_csv(train, sep=' ', header=None, index_col=0).sort_index()
        df_train.columns = ['confiavel']
        df_train = pd.concat([df, df_train], axis=1, join='inner')
        y = df_train.loc[:, 'confiavel'].values
        X = df_train.drop(['confiavel'], axis=1)
        result = None
        model = None
        if (args.model == "knn"):
                nneigh = args.nneigh
                model = KNN(n_neighbors=nneigh).fit(X,y)
                result = pd.DataFrame(model.predict(df).T, index=df.index).to_dict()[0]
        elif(args.model == "svm"):
                C = args.C
                kernel = args.kernel
                model = SVC(C=C, random_state=0, kernel=kernel, class_weight='balanced', gamma='scale').fit(X,y)
                result = pd.DataFrame(model.predict(df).T, index=df.index).to_dict()[0]
        else:
                model = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced').fit(X,y)
                result = pd.DataFrame(model.predict_proba(df).T[1], index=df.index).to_dict()[0]
        folds = 3
        score = cross_val_score(model, X,y, scoring='balanced_accuracy', cv=folds).mean()
        logging.info("Fim de classificacao da rede com {}. Balanced accuracy {} folds: {}".format(args.model, folds, score) )

        
        return result

def sybil_rank(G, init_p, p_lr_syb):
        logging.info("Iniciando propagacao de confianca")
        p_lr_syb = float(p_lr_syb)
        G = G.G
        n_iter = int(log(len(G.keys())))
        i_base = init_p
        for i in range(n_iter):
                logging.info("Propagacao de confianca iteracao {}".format(i))
                i_atual = {}
                for v in G.keys():
                        #somar os pesos dos vizinhos divididos por cada grau
                        c_vizinhos = [float(i_base[j])/len(G[j]) for j in G[v]]
                        i_atual[v] = p_lr_syb*float(i_base[v]) + (1-p_lr_syb)*np.sum(c_vizinhos)
                i_base = i_atual
        return i_base
