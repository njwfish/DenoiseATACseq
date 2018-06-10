from keras import optimizers
import collections
from sklearn.model_selection import train_test_split
import numpy as np
from keras_utils import save_model, load_model
from keras.models import Sequential
from keras.layers import *
import h5py
from nash import NASH

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

seed = 8
np.random.seed(seed)

# Load in sample of dataset to search architecture on; searching on whole dataset impractical
h5f = h5py.File('SRR891270.sorted.bam_peaks.narrowPeak.h5', 'r')
X = h5f['1lq'][:]
Y = h5f['1hq'][:]
for chrom in list(HG38_CHROM_SIZES.keys())[1:]:
    np.append(X, h5f[chrom + 'lq'][:])
    np.append(Y, h5f[chrom + 'hq'][:])
h5f.close()
X = np.reshape(X, np.shape(X) + (1,))

index = np.random.choice(len(X), len(X))
X = X[index, :, :, :]
Y = Y[index, :]

# Load and compile a baseline trained model
#model, h = load_model('test')
#model.compile(optimizer="adam", loss="mse")
model = Sequential()
model.add(InputLayer((64,64,1)))
model.add(Conv2D(12, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(1, (5, 5), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.compile(optimizer="adam", loss="mse")
model.summary()

# Run architecture search and save best result
n = NASH(100, 8, 10, X, Y)
best = n.search(model, save_nets=True)
save_model("best", best, {})