import numpy as np
import matplotlib.pyplot as plt
import pickle

# Fixing random state for reproducibility

from os import listdir
model_ids = [f for f in listdir('/Users/njwfish/Dropbox/ATACseq/his/')]
print(model_ids)
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
for model in model_ids:
    start = int(model.split('_')[1])
    x = np.arange(10 * start, 10 * (start + 1))
    with open('/Users/njwfish/Dropbox/ATACseq/his/' + model, 'rb') as file_pi:
        try:
            history = pickle.load(file_pi)
        except:
            continue
    y = history['loss']
    if len(y) != len(x) or max(y) > 300:
        continue
    ax1.scatter(x, y, color=(0.0166 * start, 0.1, 0.1), s=0.2)
    ax1.set_title('MSE Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    y = history['val_loss']
    if len(y) != len(x) or max(y) > 300:
        continue
    ax2.scatter(x, y, color=(0.1, 0.0166 * start, 0.1), s=0.2)
    ax2.set_title('MSE Validation Loss')
    ax2.set_xlabel('Iterations')


plt.show()