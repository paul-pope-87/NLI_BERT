import pandas as pd

def get_SD(df, col):
    cats = df[col].shape[1]
    mean = df[col].sum(axis=1) / cats
    sos = df[col].sub(mean, axis = 0).pow(2).sum(axis=1)
    SD = sos.div(cats).pow(0.5)
    return SD
    
def get_CV(df, col):
    cats = df[col].shape[1]
    sd = get_SD(df, col)
    mean = df[col].sum(axis=1) / cats
    CV = sd.div(mean)
    return CV