import pandas as pd
import numpy as np
import json
import operator
import math,logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from math import log, floor
from sklearn.model_selection import cross_val_score,KFold

def load_df(args):
        input = args.output #resultado (output) do s2v
        train = args.train
        df = pd.read_csv(input, skiprows=1, sep=' ', header=None, index_col=0).sort_index()
        df_train = pd.read_csv(train, sep=' ', header=None, index_col=0).sort_index()
        df_train.columns = ['confiavel']
        df_train = pd.concat([df, df_train], axis=1, join='inner')
        y = df_train.loc[:, 'confiavel']
        X = df_train.drop(['confiavel'], axis=1)
        return X,y,df

def exec_class(X_fold, y_fold, df, args):
        result = None
        model = None
        if (args.model == "knn"):
                nneigh = args.nneigh
                model = KNN(n_neighbors=nneigh).fit(X_fold,y_fold)
                result = pd.DataFrame(model.predict(df).T, index=df.index).to_dict()[0]
        elif(args.model == "svm"):
                C = args.C
                kernel = args.kernel
                model = SVC(C=C, random_state=0, kernel=kernel, class_weight='balanced', gamma='scale').fit(X_fold,y_fold)
                result = pd.DataFrame(model.predict(df).T, index=df.index).to_dict()[0]
        else:
                C = args.C
                model = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced', C=C).fit(X_fold,y_fold)
                result = pd.DataFrame(model.predict_proba(df).T[1], index=df.index).to_dict()[0]
        return result

def set_classification(args, X,y, df, folds=1):
        results = []
        if (folds > 1):
                kf = KFold(n_splits=folds, shuffle=True)
                fold = 1
                for train_index, test_index in kf.split(X):
                        X_fold = X.loc[train_index].dropna()
                        y_fold = y.loc[train_index].dropna()
                        logging.info('Iniciando classificacao da rede com fold {}'.format(fold))
                        result = exec_class(X_fold, y_fold, df, args)
                        
                        logging.info("Fim de classificacao da rede com {}.".format(args.model) )
                        results.append(result)
                        fold +=1
        else:
                X_fold = X
                y_fold = y
                logging.info('Iniciando classificacao da rede')
                
                result = exec_class(X_fold, y_fold, df, args)
                
                logging.info("Fim de classificacao da rede com {}.".format(args.model) )
                results.append(result)
       
        return results

def sybil_rank(G, init_p, p_lr_syb):
        logging.info("Iniciando propagacao de confianca")
        p_lr_syb = float(p_lr_syb)
        G = G.G
        n_iter = int(log(len(G.keys()))/2)
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
