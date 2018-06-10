import h5py
import keras
from keras.models import Model
from keras.layers import *
import h5py
import numpy as np
import collections
from keras_utils import save_model, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

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


#h5f = h5py.File('SRR891270.sorted.bam_peaks.narrowPeak.h5', 'r')
#X = h5f['1lq'][:]
#Y = h5f['1hq'][:]
#for chrom in list(HG38_CHROM_SIZES.keys())[1:]:
#    np.append(X, h5f[chrom + 'lq'][:])
#    np.append(Y, h5f[chrom + 'hq'][:])
#h5f.close()
#X = np.reshape(X, np.shape(X) + (1,))

#np.random.seed(1)
#index = np.random.choice(len(X), len(X))
#X = X[index, :, :, :]
#Y = Y[index, :]

for mid in ['model_0', 'model_20', 'model_40', 'best']:
    model, h = load_model(mid)
    model.summary()
    #model.compile('adam', 'mse')
    #train = model.evaluate(X[:-972,:,:,:], Y[:-972,:])
    #test = model.evaluate(X[-972:,:,:,:], Y[-972:,:])
    #print('Train: \t' + str(train) + '\tTest: \t' + str(test))
