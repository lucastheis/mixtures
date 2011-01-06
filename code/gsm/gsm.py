"""
An implementation of the finite Gaussian scale mixture.
"""

from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov
from numpy import square, sqrt, exp, log, pi, squeeze, diag
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig
from scipy.special import gammaincinv
from scipy.optimize import bisect
from scipy.stats import chi
from utils import logsumexp

class GSM:
	"""
	Finite Gaussian scale mixture.
	"""

	def __init__(self, dim, num_scales):
		"""
		Initialize parameters and hyperparameters.
		"""

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
		"""
		Generate samples from the model.
		"""

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




	def train(self, data, posterior=None):
		"""
		Update parameters. The result depends on the previous parameters.
		"""

		if posterior is None:
			posterior = exp(self.logposterior(data))
		else:
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
		self.scales = self.dim * sum(posterior[:, :], 1) / \
		    squeeze(dot(posterior[:, :], tmp))



	def gaussianize(self, data):
		"""
		Apply radial Gaussianization.
		"""

		# inverse Gaussian radial CDF
		gicdf = lambda y: sqrt(2. * gammaincinv(self.dim / 2., y))

		# center data
		data = data - self.mean

		# whiten data
		val, vec = eig(self.precision)
		data = dot(dot(vec, dot(diag(sqrt(val)), vec.T)), data)

		# normalize data
		norm = sqrt(sum(square(data - self.mean), 0))

		# allocate memory
		result = zeros_like(norm)

		# transform data
		for j in range(self.num_scales):
			result += self.priors[j] * chi.cdf(sqrt(self.scales[j]) * norm, self.dim)

		return multiply(gicdf(result) / norm, data)



	def invgaussianize(self, data, maxiter=100):
		"""
		Apply inverse radial Gaussianization.
		"""

		def rcdf(norm):
			"""
			Radial CDF.
			"""
			return sum(self.priors * chi.cdf(sqrt(self.scales) * norm, self.dim))

		# compute norm
		norm = sqrt(sum(square(data), 0))

		# normalize data
		data = data / norm

		# apply Gaussian radial CDF
		norm = chi.cdf(norm, self.dim)

		# apply inverse radial CDF
		norm_max = 1.
		for t in range(len(norm)):
			# make sure root lies between zero and norm_max
			while rcdf(norm_max) < norm[t]:
				norm_max += 1.
			# numerically find root
			norm[t] = bisect(
			    f=lambda x: rcdf(x) - norm[t],
			    a=0.,
			    b=norm_max,
			    maxiter=maxiter,
			    disp=False)

		# apply norm
		data = multiply(norm, data)

		# unwhiten data
		val, vec = eig(self.precision)
		data = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), data)

		# shift data
		data += self.mean

		return data



	def loglikelihood(self, data):
		"""
		Calculate marginal log-likelihood with respect to the given data points.
		"""
		return logsumexp(self.logjoint(data), 0)



	def logposterior(self, data):
		"""
		Calculate log-posterior over scales given the data points.
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
