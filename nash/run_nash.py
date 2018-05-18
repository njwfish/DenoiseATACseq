from keras import optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from keras_utils import save_model, load_model
import h5py
from nash import NASH

seed = 8
np.random.seed(seed)

# Load in sample of dataset to search architecture on; searching on whole dataset impractical
X_str = 'X0.h5'
with h5py.File(X_str, 'r') as hf:
    X = hf[X_str][:]
    np.random.shuffle(X)
Y_str = 'Y0.h5'
with h5py.File(Y_str, 'r') as hf:
    Y = hf[Y_str][:]
    np.random.shuffle(Y)

# NASH implementation not set up for 1D conv as necessary here, so instead map to 2D, with 2 1D dimensions
X = np.reshape(X, np.shape(X) + (1, 1,))
Y = np.reshape(Y, np.shape(Y) + (1, 1,))

# Split data for train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)

# Load and compile a baseline trained model
model, h = load_model('baseline')
model.compile(optimizer="adam", loss="mse")
model.summary()

# Run architecture search and save best result
n = NASH(2, 2, 2, X_train, y_train)
best = n.search(model)
save_model("best", best, {})