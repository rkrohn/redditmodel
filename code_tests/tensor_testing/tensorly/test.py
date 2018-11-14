import numpy as np
import random
#from tensorly.random.noise import add_noise
#from tensorly.random import check_random_state
from tensorly.decomposition import robust_pca


np.set_printoptions(precision=4)		#no scientific notation please
np.set_printoptions(suppress=True)

#create random np array to use as tensor
x = np.random.randint(20, size=(1, 9, 7))		#number of arrays, number rows, number columns
print("\ntensor shape:", x.shape)
print("\noriginal data\n", x)

#add some random noise, just for kicks
noise = np.random.normal(0, 0.5, size=x.shape)		#random values
#print(noise)
mask = np.random.binomial(1, 0.15, size=x.shape)	#mask for those random values
#print("noise mask\n", mask)
x_noisy = x + (noise * mask)
print("\nnoisy data\n", x_noisy)

#decompose tensor into low rank (denoised) and sparse (noise)
low_rank_part, sparse_part = robust_pca(x_noisy, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
print('\nx.shape={} == low_rank_part.shape={} == sparse_part.shape={}.\n'.format(x.shape, low_rank_part.shape, sparse_part.shape))
print("denoised\n", sparse_part)		#seems backwards?

#try missing values, instead of noise - but use the same mask
missing_mask = 1 - mask
x_missing = x.astype(np.float64) * missing_mask
print("\nmissing data\n", x_missing)

#decompose again
low_rank_part, sparse_part = robust_pca(x_missing, mask=~missing_mask, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
print("\nfilled in\n", sparse_part)