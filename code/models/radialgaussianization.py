"""
An implementation of the Gaussianization transform.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov
from numpy import square, sqrt, exp, log, pi, squeeze, diag, power
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig, slogdet
from transform import Transform
from scipy.stats import chi
from scipy.special import gamma
from utils import logsumexp, gammaincinv

class RadialGaussianization(Transform):
	def __init__(self, model, gsm):
		"""
		@type  model: Distribution
		@param model: the model applied to the transformed data

		@type  gsm: GSM
		@param gsm: the model used for the Gaussianization
		"""

		self.model = model
		self.gsm = gsm



	def function(self, data):
		"""
		Radial Gaussianization.
		"""

		def rcdf(norm):
			"""
			Radial CDF.
			"""

			# allocate memory
			result = zeros_like(norm)

			for j in range(self.gsm.num_scales):
				result += self.gsm.priors[j] * grcdf(sqrt(self.gsm.scales[j]) * norm, self.gsm.dim)
			result[result > 1.] = 1.

			return result

		# center data
		data = data - self.gsm.mean

		# whiten data
		val, vec = eig(self.gsm.precision)
		data = dot(dot(vec, dot(diag(sqrt(val)), vec.T)), data)

		# compute norm
		norm = sqrt(sum(square(data), 0))

		# radial Gaussianization transform
		return multiply(igrcdf(rcdf(norm), self.gsm.dim) / norm, data)


	def inverse(self, data):
		def rcdf(norm):
			"""
			Radial CDF.

			@type  norm: float
			@param norm: one-dimensional, positive input
			"""
			return sum(self.gsm.priors * grcdf(sqrt(self.gsm.scales) * norm, self.gsm.dim))

		# compute norm
		norm = sqrt(sum(square(data), 0))

		# normalize data
		data = data / norm

		# apply Gaussian radial CDF
		norm = grcdf(norm, self.gsm.dim)

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
		val, vec = eig(self.gsm.precision)
		data = dot(dot(vec, dot(diag(1. / sqrt(val)), vec.T)), data)

		# shift data
		data += self.gsm.mean

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

			for j in range(self.gsm.num_scales):
				result += self.gsm.priors[j] * grcdf(sqrt(self.gsm.scales[j]) * norm, self.gsm.dim)
			result[result > 1.] = 1.

			return result


		def logdrcdf(norm):
			"""
			Logarithm of the derivative of the radial CDF.
			"""

			# allocate memory
			result = zeros([self.gsm.num_scales, len(norm)])

			tmp = sqrt(self.gsm.scales)

			for j in range(self.gsm.num_scales):
				result[j, :] = log(self.gsm.priors[j]) + logdgrcdf(tmp[j] * norm, self.gsm.dim) + log(tmp[j])

			return logsumexp(result, 0)

		# center data
		data = data - self.gsm.mean

		# whitening transform
		val, vec = eig(self.gsm.precision)
		whiten = dot(vec, dot(diag(sqrt(val)), vec.T))

		# whiten data
		data = dot(whiten, data)

		# log of Jacobian determinant of whitening transform
		_, logtmp3 = slogdet(self.gsm.precision)
		logtmp3 /= 2.

		# data norm
		norm = sqrt(sum(square(data), 0))

		# radial gaussianization function applied to the norm
		tmp1 = igrcdf(rcdf(norm), self.gsm.dim)

		# log of derivative of radial gaussianization function
		logtmp2 = logdrcdf(norm) - logdgrcdf(tmp1, self.gsm.dim)

		# return log of Jacobian determinant
		return (self.gsm.dim - 1) * log(tmp1 / norm) + logtmp2 + logtmp3



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
