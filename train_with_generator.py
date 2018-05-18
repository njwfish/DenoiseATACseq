import h5py
import numpy as np
from keras_utils import save_model, load_model

def generator(batch_size):
    """
    Thisis a simple generator, it cannot be parallelized, will change in later implementations
    :param batch_size:
    :return:
    """
    while True:
        for i in range(22):
            X_str = 'X' + str(i) + '.h5'
            with h5py.File(X_str, 'r') as hf:
                X = hf[X_str][:]
                np.random.shuffle(X)
            Y_str = 'X' + str(i) + '.h5'
            with h5py.File(Y_str, 'r') as hf:
                Y = hf[Y_str][:]
                np.random.shuffle(Y)
            assert X.shape == Y.shape, "Data incorrect"
            for j in range(int(len(X) / batch_size)):
                X_batch = X[j: j + batch_size, :]
                X_batch = np.reshape(X_batch, np.shape(X_batch) + (1, 1,))
                Y_batch = Y[j: j + batch_size, :]
                Y_batch = np.reshape(Y_batch, np.shape(Y_batch) + (1, 1,))
                yield X_batch, Y_batch

# Load and train baseline on data to prep for arhitecture search
batch_size = 32
model, h = load_model('baseline')
model.compile(optimizer="adam", loss="mse")
history = model.fit_generator(generator(batch_size), steps_per_epoch=65625, nb_epoch=1)
history.history = {k: h[k] + history.history[k] for k in h.keys()}
save_model('baseline', model, history)