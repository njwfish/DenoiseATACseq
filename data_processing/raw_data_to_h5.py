import csv
import numpy as np
import h5py

def csv_to_h5(tsv, prefix, prefix_cols_to_exclude):
    """
    Takes input data file and converts to batch files of 100,000 loadable as numpy arrays via h5 compression
    :param tsv: tab-seperated file of data to convert
    :param prefix: prefix for h5 files
    :param prefix_cols_to_exclude: number of non-numerical prefix columns to exclude
    """
    with open(tsv, 'r') as tsv_in:
        i = 0
        j = 0
        X = np.empty((100000, 1000), float)
        for line in csv.reader(tsv_in, delimiter='\t'):
            i += 1
            if i == 100000:
                file_str = prefix + str(j) + '.h5'
                with h5py.File(file_str, 'w') as hf:
                    hf.create_dataset(file_str, data=X)
                X = np.empty((100000, 1000), float)
                j += 1
                i = 0
            X[i, :] = np.array(line[prefix_cols_to_exclude:], dtype=float)
        file_str = prefix + str(j) + '.h5'
        with h5py.File(file_str, 'w') as hf:
            hf.create_dataset(file_str, data=X)

# Convert X and Y to numpy arrays and save
csv_to_h5('noisy_output.txt', 'X', 1)
csv_to_h5('data_output.txt', 'Y', 2)
