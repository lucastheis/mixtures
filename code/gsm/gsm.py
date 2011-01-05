from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov, eye
from numpy import square, sqrt, exp, log, pi, squeeze, diag, round, sort
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig
from scipy.special import gamma, gammainc, gammaincinv
from utils import logsumexp

class GSM:
	def __init__(self, dim, num_scales):
		self.dim = dim
		self.num_scales = num_scales

		# initial prior over scales
		self.priors = ones(num_scales) / num_scales

		# initial scale parameters
		self.scales = 0.75 + rand(num_scales) / 2.

		# initial precision matrix
		self.precision = inv(cov(randn(dim, dim * dim)))

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

		# scale samples
		return multiply(samples, 1. / sqrt(self.scales[ind])) + self.mean




	def train(self, data, posterior):
		"""
		"""

		posterior = multiply(posterior, exp(self.logposterior(data)))

		# helper variable
		weights = posterior * self.scales.reshape(-1, 1)

		# adjust prior over scales
		self.priors = mean((posterior + 0.001) / sum(posterior + 0.001, 0), 1)

		# adjust mean
		self.mean = sum(dot(data, weights.T), 1) / sum(weights)
		self.mean.resize([self.dim, 1])

		# center data
		tmp = data - self.mean

		# adjust precision matrix
		self.precision = zeros_like(self.precision)
		for j in range(self.num_scales):
			self.precision += cov(multiply(tmp, sqrt(weights[j, :])))
		self.precision = inv(self.precision)
		self.precision /= pow(det(self.precision), 1. / self.dim)

		tmp = sum(multiply(tmp, dot(self.precision, tmp)), 0)
		tmp.resize([tmp.shape[0], 1])

		# adjust scales
		self.scales = self.dim * sum(posterior[:, :], 1) / squeeze(dot(posterior[:, :], tmp))



	def gaussianize(self, data):
		# Gaussian radial CDF and inverse radial CDF
		rcdf = lambda x: gammainc(self.dim / 2., square(x) / 2.)
		icdf = lambda y: sqrt(2. * gammaincinv(self.dim / 2., y))

		# center data
		data = data - self.mean

		# whiten data
		val, vec = eig(self.precision)
		data = dot(dot(vec, dot(diag(sqrt(val)), vec.T)), data)

		norm = sqrt(sum(square(data), 0))
		data = data / norm

		# allocate memory
		result = zeros_like(norm)

		# transform data
		for j in range(self.num_scales):
			result += self.priors[j] * rcdf(sqrt(self.scales[j]) * norm)

		return multiply(icdf(result), data)



	def loglikelihood(self, data):
		return logsumexp(self.logjoint(data), 0)



	def logposterior(self, data):
		jnt = self.logjoint(data)
		return jnt - logsumexp(jnt, 0)



	def logjoint(self, data):
		"""
		Compute joint distribution over scales and data points.
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
