import numpy as np
import keras
from os import listdir
from os.path import isfile, join
import pickle
import collections
import h5py

HG38_CHROM_SIZES = collections.OrderedDict([
    ('1',   9725),
    ('10',  5227),
    ('11',  5227),
    ('12',  5207),
    ('13',  4468),
    ('14',  4182),
    ('15',  3985),
    ('16',  3529),
    ('17',  3253),
    ('18',  3140),
    ('19',  2290),
    ('2',   9461),
    ('20',  2518),
    ('21',  1825),
    ('22',  1986),
    ('3',   7746),
    ('4',   7431),
    ('5',   7092),
    ('6',   6673),
    ('7',   6225),
    ('8',   5670),
    ('9',   5407),
    ('X',   6096),
    ('Y',   2236),
])

class RandDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(64,64), y_dim=1024, n_channels=1,
                  shuffle=True):
        'Initialization'
        self.dim = dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.files = [join('data/', f) for f in listdir('data/') if isfile(join('data/', f))]
        self.chroms = list(HG38_CHROM_SIZES.keys())
        self.indexes = None
        self.curr_file = 0
        self.curr_chrom = 0
        self.curr_pos = 0
        self.curr_X = None
        self.curr_Y = None
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(43235777648 / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X[:, :, :, 0] = self.curr_X[self.indexes[self.curr_pos:self.curr_pos + self.batch_size]]
        Y = self.curr_X[self.indexes[self.curr_pos:self.curr_pos + self.batch_size]]

        self.curr_pos += self.batch_size

        if self.curr_pos >= HG38_CHROM_SIZES[self.chroms[self.curr_chrom]]:
            self.curr_pos = 0
            self.curr_chrom += 1
            if self.curr_chrom >= len(self.chroms):
                self.curr_chrom = 0
                self.curr_file += 1
                if self.curr_file >= len(self.files):
                    self.on_epoch_end()
            self.__update_curr_data()

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.curr_file = 0
        self.curr_chrom = 0
        self.curr_pos = 0
        if self.shuffle == True:
            np.random.shuffle(self.files)
            np.random.shuffle(self.chroms)
            self.indexes = np.arange(HG38_CHROM_SIZES[self.chroms[self.curr_chrom]])
            np.random.shuffle(self.indexes)
        self.__update_curr_data()

    def __update_curr_data(self):
        h5f = h5py.File(self.files[self.curr_file], 'r')
        self.curr_X = h5f[self.chroms[self.curr_chrom] + 'lq'][:]
        self.curr_Y = h5f[self.chroms[self.curr_chrom] + 'hq'][:]
        h5f.close()
