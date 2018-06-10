import h5py
import numpy as np
from keras_utils import save_model, load_model
from generator import RandDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Reshape, Dense, GRU, Flatten
import h5py
import numpy as np
import collections
from keras_utils import save_model, load_model

# Load and train baseline on data to prep for arhitecture search
batch_size = 32
model = Sequential()
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', input_shape=(64, 64, 1)))
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', input_shape=(64, 64, 1)))
model.add(Reshape((32, 1024)))
model.add(GRU(1024, activation='relu', reset_after=True))
model.add(Dense(10, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.compile(optimizer="adam", loss="mse")
history = model.fit_generator(RandDataGenerator(batch_size), verbose=1, steps_per_epoch=100, epochs=1)
#history.history = {k: h[k] + history.history[k] for k in h.keys()}
save_model('test', model, history)