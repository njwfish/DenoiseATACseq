from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from keras.models import model_from_yaml
from keras.layers import Conv2D
from keras import regularizers
from keras import optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from random import randint
from data_processing import load_processed_data
import os
import pickle



def gen_rand_architecture():
    """
    Create random model architecture
    """
    conv_layers = randint(1, 4)
    filters_conv_layer = np.random.randint(5, 11, conv_layers)
    filter_size_conv_layer = np.random.randint(4, 11, conv_layers)
    # stride_size_conv_layer = np.random.randint(1, 5, conv_layers)
    dense_layers = randint(0, 3)
    units_dense_layer = np.random.randint(3, 8, dense_layers)
    model_id = str(conv_layers) + "".join(map(str, filters_conv_layer)) + "".join(map(str, filter_size_conv_layer)) \
               + str(dense_layers) + "".join(map(str, units_dense_layer))
    architecture = conv_layers, filters_conv_layer, filter_size_conv_layer, dense_layers, units_dense_layer
    return model_id, architecture


def r2(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def compile_model(conv_layers, filters_conv_layer, filter_size_conv_layer, dense_layers, units_dense_layer):
    """
    Build random sequential model from blueprint
    """
    model = Sequential()
    model.add(InputLayer((150, 4, 1)))
    for i in range(conv_layers):
        model.add(Conv2D(filters_conv_layer[i], (filter_size_conv_layer[i], 4),
                         #kernel_regularizer=regularizers.l2(0.01),
                         padding="same", activation="relu"))
    model.add(Flatten())
    for i in range(dense_layers):
        model.add(Dense(units_dense_layer[i], activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(6, activation="relu"))#, kernel_regularizer=regularizers.l2(0.01)))
    adam = optimizers.adam(amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', r2])
    return model


def save_model(model_id, model, history):
    if history != {}:
        with open('models/' + model_id + '.hist', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    model_yaml = model.to_yaml()
    with open('models/' + model_id + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights('models/' + model_id + ".h5")


def load_model(model_id):
    with open('models/' + model_id + '.hist', 'rb') as file_pi:
        history = pickle.load(file_pi)
    with open('models/' + model_id + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    model.load_weights('models/' + model_id + ".h5")
    return model, history


def rand_architecture_search(X_train, X_test, y_train, y_test):
    """
    Build and train random architectures
    """
    while True:
        trained_models = [f.split(".")[0] for f in os.listdir('models/')]
        model_id, architecture = gen_rand_architecture()
        while model_id in trained_models:
            model_id, architecture = gen_rand_architecture()

        model = compile_model(*architecture)
        history = model.fit(X_train, y_train, validation_split=0.05, epochs=100, batch_size=32, verbose=0)
        save_model(model_id, model, history)

        score = model.evaluate(X_test, y_test, verbose=0)
        print(model_id + "%s: %.2f" % (model.metrics_names[1], score[1]))


def train_in_depth(model_id):
    model, h = load_model(model_id)
    adam = optimizers.adam(amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae', r2])
    score = model.evaluate(X_test, y_test, verbose=2)
    print("%s: %.2f" % (model.metrics_names[1], score[1]))
    history = model.fit(X_train, y_train, validation_split=0.05, epochs=1000, batch_size=32, verbose=2)
    history.history = {k: h[k] + history.history[k] for k in h.keys()}
    save_model(model_id, model, history)
    score = model.evaluate(X_test, y_test, verbose=2)
    print("%s: %.2f" % (model.metrics_names[1], score[1]))

def set_lambda(model_id, model, l):
    """
    Add regularization term to model
    """
    for i in range(l):
        model.layers[i+1].kernel_regularizer = regularizers.l2(0.00001)
    with open('models/' + model_id + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
