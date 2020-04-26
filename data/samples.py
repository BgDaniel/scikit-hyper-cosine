import pandas as pd
import numpy as np
import os

PATH_TO_SIMPLICES = 'simplices.csv'

def get_data(*args, target):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DATA_FILE = os.path.join(dir_path, PATH_TO_SIMPLICES)
    samples = pd.read_csv(DATA_FILE, sep=';')

    if(args == None):
        return samples
    else:
        columns = []

        for arg in args:
            columns.append(samples[arg].values)

        return  np.column_stack(tuple(columns)), samples[target].values
