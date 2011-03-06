"""
An implementation of the multivariate Bernoulli distribution.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy.random import rand
from numpy import sum, mean, asarray, prod, log, dot, zeros_like, seterr
from distribution import Distribution

class Bernoulli(Distribution):
	"""
	Multivariate Bernoulli distribution.
	"""

	def __init__(self, dim):
		self.dim = dim
		self.pvalues = rand(dim, 1)
		self.pvalues = (self.pvalues + 1.) / 3.

		# regularization parameter
		self.alpha = 10.



	def sample(self, num_samples=1):
		return (rand(self.dim, num_samples) < self.pvalues) * 1



	def train(self, data, weights=None):
		# make sure data is stored in a NumPy array
		data = asarray(data)

		# used for regularization
		epsilon = 0.

		if weights is None:
			self.pvalues = mean(data, 1)
			if self.alpha is not None:
				epsilon = float(self.alpha) / data.shape[1]
		else:
			self.pvalues = dot(data, weights) / sum(weights)
			if self.alpha is not None:
				epsilon = float(self.alpha) / sum(weights)
		self.pvalues = (self.pvalues + epsilon) / (1. + 2. * epsilon)
		self.pvalues = self.pvalues.reshape(-1, 1)



	def loglikelihood(self, data):
		# make sure data is stored in a NumPy array
		data = asarray(data)

		# compute likelihood
		lik = prod(data * self.pvalues + (1. - data) * (1. - self.pvalues), 0)

		# do not return -inf
		zero = (lik == 0.)
		loglik = zeros_like(lik)
		loglik[-zero] = log(lik[-zero])
		loglik[zero] = -1E300

		return loglik
