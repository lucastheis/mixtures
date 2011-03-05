"""
An implementation of the multivariate Bernoulli distribution.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy.random import rand
from numpy import sum, mean, asarray, prod
from distribution import Distribution

class Bernoulli(Distribution):
	"""
	Multivariate Bernoulli distribution.
	"""

	def __init__(self, dim):
		self.dim = dim
		self.pvalues = rand(dim, 1)



	def sample(self, num_samples=1):
		return (rand(self.dim, num_samples) < self.pvalues) * 1



	def train(self, data, weights=None):
		# make sure data is stored in a NumPy array
		data = asarray(data)

		if weights is None:
			self.pvalues = mean(data, 1)
		else:
			self.pvalues = mean(weights * data, 1) / mean(weights, 1)



	def loglikelihood(self, data):
		# make sure data is stored in a NumPy array
		data = asarray(data)

		return prod(self.pvalues * data + (1. - self.pvalues) * (1 - data), 0)
