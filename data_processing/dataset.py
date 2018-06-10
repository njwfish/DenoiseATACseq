import csv 
import numpy as np
from math import ceil
import h5py
import collections
import sys

HG38_CHROM_SIZES = collections.OrderedDict([
    ('1',   248956422),
    ('10',  133797422), 
    ('11',  135086622), 
    ('12',  133275309), 
    ('13',  114364328), 
    ('14',  107043718), 
    ('15',  101991189), 
    ('16',  90338345),  
    ('17',  83257441),  
    ('18',  80373285),  
    ('19',  58617616),  
    ('2',   242193529),
    ('20',  64444167),  
    ('21',  46709983),  
    ('22',  50818468),  
    ('3',   198295559), 
    ('4',   190214555), 
    ('5',   181538259), 
    ('6',   170805979), 
    ('7',   159345973), 
    ('8',   145138636), 
    ('9',   138394717), 
    ('X',   156040895), 
    ('Y',   57227415),  
])

HILBERT_NUM = 64
HILBERT_SQUARE = HILBERT_NUM * HILBERT_NUM
BIN_SIZE = 25
CHUNK_BINS = 1024
CHUNK_SIZE = BIN_SIZE * CHUNK_BINS

def tsv_to_list(tsv):
    with open(tsv, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

def assign_signal(peaks, i, chrom, chrom_len):
    X = np.zeros((ceil(chrom_len/CHUNK_SIZE) * CHUNK_BINS, 1))
    while i < len(peaks) and peaks[i][0] == chrom:
        start = int(peaks[i][1])
        end = int(peaks[i][2])
        signal = int(peaks[i][4])
        bin_start = int(start/BIN_SIZE)
        bin_end = ceil(end/BIN_SIZE)
        X[bin_start:bin_end] += signal
        i += 1
    return X, peaks, i

def gen_out(hq_peaks_file, lq_peaks_file):
    hq_peaks = tsv_to_list(hq_peaks_file)
    lq_peaks = tsv_to_list(lq_peaks_file)
    i_hq, i_lq = 0, 0
    h5f = h5py.File(hq_peaks_file + '.h5', 'w')
    for chrom in HG38_CHROM_SIZES.keys():
        try:
            while hq_peaks[i_hq][0] != chrom:
                i_hq += 1
            while lq_peaks[i_lq][0] != chrom:
                i_lq += 1
        except:
            print("No chrom " + chrom)
            continue
        chrom_len = HG38_CHROM_SIZES[chrom]
        X_hq, peaks, i_hq = assign_signal(hq_peaks, i_hq, chrom, chrom_len)
        X_hq = X_hq.reshape((ceil(chrom_len/CHUNK_SIZE), CHUNK_BINS))
        h5f.create_dataset(chrom + 'hq', data=X_hq)
        X_lq, peaks, i_lq = assign_signal(lq_peaks, i_lq, chrom, chrom_len)
        X_lq = hilbert_reshape(X_lq, ceil(chrom_len/CHUNK_SIZE))
        h5f.create_dataset(chrom + 'lq', data=X_lq)
    h5f.close()

def hilbert_reshape(X_in, len):
    surr_space = int((HILBERT_SQUARE - CHUNK_BINS) / 2)
    pad = np.zeros((surr_space, 1))
    X_in = np.vstack((pad, X_in, pad))
    X = np.zeros((len, HILBERT_NUM, HILBERT_NUM))
    idx = hilbert_curve(HILBERT_NUM)
    for i in range(len):
        tmp = X_in[i * CHUNK_BINS : (i + 4) * CHUNK_BINS]
        X[i, :, :] = np.squeeze(tmp[idx])
    return X

def hilbert_curve(n):
    ''' Generate Hilbert curve indexing for (n, n) array. 'n' must be a power of two. '''
    # recursion base
    if n == 1:  
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]])) 

_, hq_file, lq_file = sys.argv
gen_out(hq_file, lq_file)

