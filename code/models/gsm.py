"""
An implementation of the finite Gaussian scale mixture.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov
from numpy import sqrt, exp, log, pi, squeeze, diag, eye
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig, slogdet
from distribution import Distribution
from utils import logsumexp

class GSM(Distribution):
	"""
	Finite Gaussian scale mixture.

	For regularization, use the model parameters alpha, beta, gamma and theta.
	To turn regularization off, set these parameters to None. The parameters are
	initialized with moderate values. For stronger regularization, turn alpha
	and beta up and gamma and theta down.

	B{References:}
		- M. Wainwright and E. Simoncelli (2000). I{Scale Mixtures of Gaussians
		and the Statistics of Natural Images.} NIPS 12.

	@type dim: integer
	@ivar dim: dimensionality of the model

	@type priors: array_like
	@ivar priors: prior weights over scale components

	@type mean: array_like
	@ivar mean: mean of the distribution

	@type precision: array_like
	@ivar precision: precision matrix

	@type scales: array_like
	@ivar scales: precision scale factors

	@type alpha: real > 0 or None
	@ivar alpha: parameter of Dirichlet prior over component weights

	@type gamma: real > 0 or None
	@ivar gamma: parameter of Wishart prior over the precision matrix

	@type beta: real > -1 or None
	@ivar beta: parameter of Gamma prior over precision scale factors

	@type theta: real > 0 or None
	@ivar theta: parameter of Gamma prior over precision scale factors
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

		# parameter of regularizing Dirichlet prior over priors
		self.alpha = 1.001

		# parameter of regularizing Wishart prior over precision matrix
		self.gamma = 1E2

		# parameters of regularizing Gamma prior over scales
		self.beta = 0.5
		self.theta = 1E2



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
		# compute posterior over scales (E)
		posterior = exp(self.logposterior(data))

		if weights is not None:
			# compute posterior over model and scales
			posterior *= weights

		# helper variable
		tmp1 = multiply(posterior, self.scales.reshape(-1, 1))

		# adjust prior over scales (M)
		self.priors = mean(posterior, 1)

		if self.alpha is not None:
			# regularization with Dirichlet prior
			self.priors += self.alpha - 1.
		self.priors /= sum(self.priors)

		# adjust mean (M)
		self.mean = sum(dot(data, tmp1.T), 1) / sum(tmp1)
		self.mean.resize([self.dim, 1])

		# center data
		data = data - self.mean

		# compute covariance
		covariance = zeros_like(self.precision)
		for j in range(self.num_scales):
			covariance += cov(multiply(data, sqrt(tmp1[j, :])))

		if self.gamma is not None:
			# regularization with Wishart prior
			covariance += eye(covariance.shape[0]) / self.gamma

		# compute precision matrix and normalize by determinant (M)
		self.precision = inv(covariance)
		self.precision /= exp(slogdet(self.precision)[1] / self.dim)

		# adjust scales (M)
		tmp2 = sum(multiply(data, dot(self.precision, data)), 0)
		tmp2.resize([tmp2.shape[0], 1])

		if self.theta is None or self.beta is None:
			# no regularization
			tmp3 = self.dim * sum(posterior, 1)
			tmp4 = squeeze(dot(posterior, tmp2))
		else:
			# regularization with Gamma prior
			tmp3 = self.dim * mean(posterior, 1) + 2. * self.beta
			tmp4 = squeeze(dot(posterior, tmp2)) / posterior.shape[1] + 2. / self.theta

		self.scales = tmp3 / tmp4



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
