import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf

import sys; sys.path.append('..')
from ktensor import KruskalTensor

mat = loadmat('../data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

# T = KruskalTensor(X.shape, rank=3, regularize=0.0, init='nvecs', X_data=X)
# X_predict = T.train_als_early(X, tf.train.AdadeltaOptimizer(0.05), epochs=30000)
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=30000)


np.save('../data/bread/X_cp.npy', X_predict)
