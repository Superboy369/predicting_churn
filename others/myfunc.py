import matplotlib.pyplot as plt
import pandas as pd

def extraction_integration(data):
    data_USER_ID = data.loc[:,'USER_ID']
    print(data_USER_ID)


def mmean(str,data2,temp):
    a = temp[str][0]
    b = temp[str][1]
    c = temp[str][2]
    data2.loc[data2.shape[0] - 1, 'ACCT_FEE'] = (a + b + c) / 3
