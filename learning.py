import pandas as pd
import numpy as np
import json
import operator


def set_classification(input, train):
        df = pd.read_csv(input, skiprows=1, sep=' ', header=None, index_col=0).sort_index()
        df_train = pd.read_csv(train, sep=' ', header=None, index_col=0).sort_index()
        df_train.columns = ['confiavel']
        df_train = pd.concat([df, df_train], axis=1, join='inner')
        print(df_train.head(10))

def sybil_rank():
        return
