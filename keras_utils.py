from keras.models import model_from_yaml
import pickle


def save_model(model_id, model, history):
    #with open('models/' + model_id + '.hist', 'wb') as file_pi:
    #    pickle.dump(history.history, file_pi)
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

