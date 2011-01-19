"""
An implementation of the finite Gaussian scale mixture.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov
from numpy import sqrt, exp, log, pi, squeeze, diag
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig
from distribution import Distribution
from utils import logsumexp

from numpy import sort

class GSM(Distribution):
	"""
	Finite Gaussian scale mixture.
	"""

	def __init__(self, dim, num_scales):
		"""
		Initialize parameters and hyperparameters.

		@type  dim: integer
		@param dim: dimensionality of the distribution

		@type  num_scales: integer
		@param num_scales: number of mixture components
		"""

		self.dim = dim
		self.num_scales = num_scales

		# initial prior over scales
		self.priors = ones(num_scales) / num_scales

		# initial scale parameters
		self.scales = 0.75 + rand(num_scales) / 2.

		# initial precision matrix
		self.precision = inv(cov(randn(dim, dim * dim)))
		self.precision /= pow(det(self.precision), 1. / self.dim)

		# initial mean
		self.mean = zeros([dim, 1])

		# parameter of regularizing Dirichlet prior
		self.alpha = 1.001



	def sample(self, num_samples=1):
		# draw basic samples
		val, vec = eig(self.precision)
		samples = randn(self.dim, num_samples)
		samples = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), samples)

		# sample indices from prior
		cum = 0.
		uni = rand(num_samples)
		ind = zeros(uni.shape, 'int')
		for j in range(1, self.num_scales):
			cum += self.priors[j - 1]
			ind[uni > cum] = j

		# scale and shift samples
		return multiply(samples, 1. / sqrt(self.scales[ind])) + self.mean




	def train(self, data, weights=None):
		# compute posterior over scales
		posterior = exp(self.logposterior(data))

		print '004'

		# incorporate conditional model prior
		if weights is not None:
			posterior *= weights

		print '005'

		# helper variable
		tmp1 = multiply(posterior, self.scales.reshape(-1, 1))

		print '006'

		# adjust prior over scales
		self.priors = mean((posterior + 0.001) / sum(posterior + 0.001, 0), 1)

		print '007'

		# adjust mean
		self.mean = sum(dot(data, tmp1.T), 1) / sum(tmp1)
		self.mean.resize([self.dim, 1])

		print '008'

		# center data
		data = data - self.mean

		print '009'

		# adjust precision matrix
		covariance = zeros_like(self.precision)
		for j in range(self.num_scales):
			covariance += cov(multiply(data, sqrt(tmp1[j, :])))

		print '010'

		val, vec = eig(covariance)
		val = exp(mean(log(val)) - log(val))

		self.precision = dot(vec, dot(diag(val), vec.T))

#		self.precision = inv(self.precision)


		# normalize by determinant
#		self.precision /= pow(det(self.precision), 1. / self.dim)

		print '011'

		data = sum(multiply(data, dot(self.precision, data)), 0)
		data.resize([data.shape[0], 1])

		print '012'

		# adjust scales
		self.scales = self.dim * sum(posterior[:, :], 1) / \
		    squeeze(dot(posterior[:, :], data))



	def loglikelihood(self, data):
		return logsumexp(self.logjoint(data), 0)



	def logposterior(self, data):
		"""
		Calculate log-posterior over scales given the data points.

		@type  data: array_like
		@param data: data stored in columns
		"""

		jnt = self.logjoint(data)
		return jnt - logsumexp(jnt, 0)



	def logjoint(self, data):
		"""
		Calculate log-joint density of scales and data points.
		"""

		tmp = data - self.mean
		tmp = sum(multiply(tmp, dot(self.precision, tmp)), 0)
		logptf = log(det(self.precision)) - log(2. * pi) * self.dim
		logjnt = zeros([self.num_scales, data.shape[1]])

		# compute joint distribution over scales and data points
		for j in range(self.num_scales):
			logjnt[j, :] = -0.5 * (self.scales[j] * tmp \
			    - log(self.scales[j]) * self.dim - logptf) \
			    + log(self.priors[j])

		return logjnt
