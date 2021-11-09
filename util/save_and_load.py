import pickle
import numpy as np


def load_pkl_data(file_path):
    return np.load(file_path, allow_pickle=True)


def savePKLdata(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    print(f'Data saved as pkl: {file}')
    return file
