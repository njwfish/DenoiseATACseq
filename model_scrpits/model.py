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


h5f = h5py.File('SRR891270.sorted.bam_peaks.narrowPeak.h5', 'r')
X = h5f['1lq'][:]
Y = h5f['1hq'][:]
for chrom in list(HG38_CHROM_SIZES.keys())[1:]:
    np.append(X, h5f[chrom + 'lq'][:])
    np.append(Y, h5f[chrom + 'hq'][:])
h5f.close()
X = np.reshape(X, np.shape(X) + (1,))

np.random.seed(1)
index = np.random.choice(len(X), len(X))
X = X[index, :, :, :]
Y = Y[index, :]

#inp = Input((64, 64, 1))
#z = Conv2D(66, (4, 4), activation='relu', padding='same')(inp)
#z = Conv2D(40, (4, 4), activation='relu', padding='same')(z)
#concat = concatenate([inp, z])
#z = Conv2D(13, (4, 4), activation='relu', padding='same')(concat)
#z = Flatten()(z)
#z = Dense(9, activation='relu')(z)
#z = Dense(11, activation='relu')(z)
#z = Dense(1024, activation='relu')(z)
#model = Model(inputs=inp, outputs=z)
#model.compile(optimizer="adam", loss="mse")
model, h = load_model('conn')
model.summary()

filepath = "models/conn-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='mean_squared_error', optimizer='adam')
history = parallel_model.fit(X, Y, epochs=10, batch_size=64, verbose=2, validation_split=0.05)
history.history = {k: h[k] + history.history[k] for k in h.keys()}
for i in range(10):
    history_i = parallel_model.fit(X, Y, epochs=100, batch_size=64, verbose=2, validation_split=0.05)
    history_i.history = {k: history.history[k] + history_i.history[k] for k in h.keys()}
    save_model(str(i) + '_conn', model, history_i)