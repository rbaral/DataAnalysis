'''
utility method to load data
'''
import pandas as pd

'''
loads the data file
'''
def loadData(filePath):
    df = pd.read_csv(filePath)
    return df

