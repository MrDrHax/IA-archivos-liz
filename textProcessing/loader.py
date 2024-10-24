import pandas as pd

def loadData(path)-> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def getTag(val:str) ->  str:
    if "neg" in val:
        return "negative"
    elif "pos" in val:
        return "positive"
    else:
        return "neutral"

def getData(path):
    data = loadData(path)

    X_raw = data.iloc[:, 3]
    Y_raw = data.iloc[:, 2]

    Y = [i for i in Y_raw]

    return X_raw, Y