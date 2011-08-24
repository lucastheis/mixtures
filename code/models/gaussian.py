"""
An implementation of the multivariate Gaussian distribution.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import cov, mean, zeros, sum, multiply, pi, log, dot, sqrt, diag
from numpy import real, eye, var
from numpy.linalg import inv, slogdet, eig, cholesky
from numpy.random import randn
from distribution import Distribution

class Gaussian(Distribution):
	"""
	Multivariate Gaussian distribution.
	"""

	def __init__(self, dim):
		self.dim = dim
		self.mean = zeros([dim, 1])
		self.precision = eye(dim)



	def sample(self, num_samples=1):
		# draw white samples
		samples = randn(self.dim, num_samples)

		if self.dim > 1:
			# unwhiten samples
			val, vec = eig(self.precision)
			samples = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), samples)
		else:
			samples = samples / sqrt(self.precision)

		# shift samples
		return samples + self.mean



	def initialize(self, data):
		# calculate mean and precision of data
		data_mean = mean(data, 1).reshape([self.dim, 1])
		data_cov = cov(data)

		if self.dim > 1:
			# Cholesky factor
			chol = cholesky(data_cov)

			# randomize parameters
			self.mean = data_mean + dot(chol / 4., randn(self.dim, 1))
			self.precision = inv(cov(dot(chol, randn(self.dim, self.dim * self.dim))))
		else:
			# randomize parameters
			self.mean = data_mean + sqrt(data_cov) / 4. * randn()
			self.precision = 1. / var(sqrt(data_cov) * randn(4))



	def train(self, data, weights=None):
		if weights is None:
			# adjust  mean and precision matrix
			self.mean = mean(data, 1).reshape([self.dim, 1])
			self.precision = inv(cov(data)) if self.dim > 1 else 1. / var(data)
		else:
			# adjust mean
			tmp1 = mean(weights)
			self.mean = mean(multiply(data, weights), 1) / tmp1
			self.mean.resize(self.dim, 1)

			# adjust precision matrix
			tmp2 = multiply(data - self.mean, sqrt(weights) / sqrt(data.shape[1]))
			if self.dim > 1:
				self.precision = inv(dot(tmp2, tmp2.T) / tmp1)
			else:
				self.precision = 1. / (dot(tmp2, tmp2.T) / tmp1)



	def loglikelihood(self, data):
		data = data - self.mean

		if self.dim > 1:
			_, logdet = slogdet(self.precision)
		else:
			logdet = log(self.precision)

		tmp1 = (logdet - log(2. * pi) * self.dim) / 2.
		tmp2 = sum(multiply(data, dot(self.precision, data)), 0) / 2.

		return tmp1 - tmp2
