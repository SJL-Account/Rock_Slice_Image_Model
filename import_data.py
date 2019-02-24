import pandas as pd


def import_data(path):

    data=pd.read_csv(path)

    for col in data.columns:
        print ('正在处理',col)
        data[col].astype('uint8')
    return data


