import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras_utils import load_model

def hilbert_curve(n):
    ''' Generate Hilbert curve indexing for (n, n) array. 'n' must be a power of two. '''
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))

m, h = load_model('best')

h5f = h5py.File('SRR891270.sorted.bam_peaks.narrowPeak.h5', 'r')
X = h5f['1lq'][:]
Y = h5f['1hq'][:]
for chrom in ['1']:
    np.append(X, h5f[chrom + 'lq'][:])
    np.append(Y, h5f[chrom + 'hq'][:])
h5f.close()
X = np.reshape(X, np.shape(X) + (1,))

k = []
for i in range(X.shape[0]):
    if np.sum(np.sum(X[i,:,:,:])) > 2000:
        k.append(i)

print(k)
c=k[-15]
idx = hilbert_curve(64)
X = X[c, :, :, :]
X = np.reshape(X, (1,) + np.shape(X))
Y = Y[c, :]
print(X.shape)
ypred = m.predict(X)
X = X.ravel()[np.argsort(idx.ravel())]


print(ypred.shape)
f, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True)
print(X.shape)
print(Y.shape)
print(ypred.shape)
ax1.bar(np.arange(4096), np.squeeze(X))

ax1.set_xlabel('Relative Chromosome Position')
ax1.set_ylabel('Score')
ax1.set_title('Low Quality Peaks')
plt.show()

