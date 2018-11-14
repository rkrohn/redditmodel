import numpy as np
import tensorflow as tf
from scipy.io.matlab import loadmat
from ktensor import KruskalTensor
import sys

# Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('brod.mat')
X = mat['X'].reshape([10,11,8])

X[0][0][0] = np.nan
X[0][5][3] = np.nan
X[0][7][6] = np.nan
print(X[0])

# Build ktensor and learn CP decomposition using ALS with specified optimizer
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
print(T)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=20000)
print(X_predict[0])

# Save reconstructed tensor to file
np.save('X_predict.npy', X_predict)