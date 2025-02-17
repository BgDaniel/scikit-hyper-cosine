import pandas as pd
import numpy as np
import os

def get_data(file_name, *args, target):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DATA_FILE = os.path.join(dir_path, file_name)
    samples = pd.read_csv(DATA_FILE, sep=';')
    samples_truncated = samples.copy()
    
    columns = samples.head()
    for column in columns:
        if column not in args and column != target:
            samples_truncated = samples_truncated.drop(column, axis=1)

    if(args == None):
        return samples
    else:
        columns = []

        for arg in args:
            columns.append(samples[arg].values)

        return  samples_truncated, np.column_stack(tuple(columns)), samples[target].values
