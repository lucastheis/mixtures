"""
An implementation of the finite Gaussian scale mixture.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov
from numpy import square, sqrt, exp, log, pi, squeeze, diag, power
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig, slogdet
from scipy.special import gammaincinv, gamma
from scipy.optimize import bisect
from scipy.stats import chi
from utils import logsumexp

from numpy import arange, min, max
from matplotlib.pyplot import plot

class GSM:
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

		def rcdf(norm):
			"""
			Radial CDF.
			"""

			# allocate memory
			result = zeros_like(norm)

			for j in range(self.num_scales):
				result += self.priors[j] * grcdf(sqrt(self.scales[j]) * norm, self.dim)

			return result

		# center data
		data = data - self.mean

		# whiten data
		val, vec = eig(self.precision)
		data = dot(dot(vec, dot(diag(sqrt(val)), vec.T)), data)

		# compute norm
		norm = sqrt(sum(square(data - self.mean), 0))

		# radial Gaussianization transform
		return multiply(igrcdf(rcdf(norm), self.dim) / norm, data)



	def invgaussianize(self, data, maxiter=100):
		"""
		Apply inverse radial Gaussianization.
		"""

		def rcdf(norm):
			"""
			Radial CDF.

			@type  norm: float
			@param norm: one-dimensional, positive input
			"""
			return sum(self.priors * grcdf(sqrt(self.scales) * norm, self.dim))

		# compute norm
		norm = sqrt(sum(square(data), 0))

		# normalize data
		data = data / norm

		# apply Gaussian radial CDF
		norm = grcdf(norm, self.dim)

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

		# inverse radial Gaussianization
		data = multiply(norm, data)

		# unwhiten data
		val, vec = eig(self.precision)
		data = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), data)

		# shift data
		data += self.mean

		return data



	def logjacobian(self, data):
		"""
		Returns the logarithm of the Jacobian determinant for the
		Gaussianization transform.
		"""

		def rcdf(norm):
			"""
			Radial CDF.
			"""

			# allocate memory
			result = zeros_like(norm)

			for j in range(self.num_scales):
				result += self.priors[j] * grcdf(sqrt(self.scales[j]) * norm, self.dim)

			return result


		def logdrcdf(norm):
			"""
			Logarithm of the derivative of the radial CDF.
			"""

			# allocate memory
			result = zeros([self.num_scales, len(norm)])

			tmp = sqrt(self.scales)

			for j in range(self.num_scales):
				result[j, :] = log(self.priors[j]) + logdgrcdf(tmp[j] * norm, self.dim) + log(tmp[j])

			return logsumexp(result, 0)

		# center data
		data = data - self.mean

		# whitening transform
		val, vec = eig(self.precision)
		whiten = dot(vec, dot(diag(sqrt(val)), vec.T))

		# whiten data
		data = dot(whiten, data)

		# log of Jacobian determinant of whitening transform
		_, logtmp3 = slogdet(self.precision)
		logtmp3 /= 2.

		# data norm
		norm = sqrt(sum(square(data), 0))

		# radial gaussianization function applied to the norm
		tmp1 = igrcdf(rcdf(norm), self.dim)

		# log of derivative of radial gaussianization function
		logtmp2 = logdrcdf(norm) - logdgrcdf(tmp1, self.dim)

		# return log of Jacobian determinant
		return (self.dim - 1) * log(tmp1 / norm) + logtmp2 + logtmp3



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



def grcdf(norm, dim):
	"""
	Gaussian radial CDF.
	"""

	return chi.cdf(norm, dim)



def igrcdf(norm, dim):
	"""
	Inverse Gaussian radial CDF.
	"""

	return sqrt(2.) * sqrt(gammaincinv(dim / 2., norm))



def logigrcdf(norm, dim):
	"""
	Logarithm of the inverse Gaussian radial CDF.
	"""

	return (log(gammaincinv(dim / 2., norm)) + log(2)) / 2.



def dgrcdf(norm, dim):
	"""
	Derivative of the Gaussian radial CDF.
	"""

	tmp = square(norm) / 2.
	return power(tmp, dim / 2. - 1.) / exp(tmp) / gamma(dim / 2) * norm



def logdgrcdf(norm, dim):
	"""
	Logarithm of the derivative of the Gaussian radial CDF.
	"""

	tmp = square(norm) / 2.
	return (dim / 2. - 1.) * log(tmp) - tmp - log(gamma(dim / 2)) + log(norm)
