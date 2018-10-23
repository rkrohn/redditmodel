import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn; seaborn.set_style('whitegrid')
import numpy
from pomegranate import *

#given data, fit to a mixture model of gamma distributions
#data must be an x*1 numpy matrix
def fit_gamma_mixture(data, plot = True):
	return 0
#end fit_gamma_mixture


#numpy.set_printoptions(suppress=True)

numpy.random.seed(0)
X = numpy.random.gamma(2, 1, size=(500, 1))
X[::2] += 3
model = GeneralMixtureModel.from_samples(NormalDistribution, 2, X)

x = numpy.arange(0, 10, .01)
plt.figure(figsize=(8, 4))
plt.hist(X, bins=50, normed=True)
plt.plot(x, model.distributions[0].probability(x), label="Distribution 1")
plt.plot(x, model.distributions[1].probability(x), label="Distribution 2")
plt.plot(x, model.probability(x), label="Mixture")
plt.legend(fontsize=14, loc=2)
plt.savefig("test_fit.png")
print(model)


#fit a single gamma distribution
'''
X = numpy.random.gamma(2, 1, size=(500, 1))

x = numpy.arange(0, 10, .01)

model = GammaDistribution.from_samples(X.reshape((500,)))

plt.figure(figsize=(8, 4))
plt.hist(X, bins=50, normed=True)
plt.plot(x, model.probability(x), label="fit")
plt.legend(fontsize=14, loc=2)
plt.savefig("gamma.png")
'''