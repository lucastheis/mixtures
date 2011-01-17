"""
An implementation of the multivariate Gaussian distribution.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import cov, mean, zeros, sum, multiply, pi, log, dot, sqrt, diag
from numpy.linalg import inv, slogdet, eig
from numpy.random import randn
from distribution import Distribution

class Gaussian(Distribution):
	"""
	Multivariate Gaussian distribution.
	"""

	def __init__(self, dim):
		self.dim = dim
		self.mean = zeros([dim, 1])

		# draw precision matrix from inverse Wishart distribution
		self.precision = inv(cov(randn(dim, dim * dim)))



	def sample(self, num_samples=1):
		# draw white samples
		samples = randn(self.dim, num_samples)

		# unwhiten samples
		val, vec = eig(self.precision)
		samples = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), samples)

		# shift samples
		return samples + self.mean




	def train(self, data, weights=None):
		if weights is None:
			# adjust  mean and precision matrix
			self.mean = mean(data, 1).reshape([self.dim, 1])
			self.precision = inv(cov(data))
		else:
			# adjust mean
			tmp1 = mean(weights)
			self.mean = mean(multiply(data, weights)) / tmp1

			# adjust precision matrix
			tmp2 = multiply(data - self.mean, sqrt(weights))
			self.precision = inv(cov(tmp2) / tmp1)



	def loglikelihood(self, data):
		data = data - self.mean

		_, logdet = slogdet(self.precision)

		tmp1 = (logdet - log(2. * pi) * self.dim) / 2.
		tmp2 = sum(multiply(data, dot(self.precision, data)), 0) / 2.

		return tmp1 - tmp2
