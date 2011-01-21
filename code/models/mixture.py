"""
A generic mixture class with an implementation of EM.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import multiply, dot, sum, mean, cov, sqrt, log, exp, pi, argsort
from numpy import ones, zeros, zeros_like, eye, round, squeeze, concatenate
from numpy.random import multinomial, rand, permutation
from numpy.linalg import det, inv, eig
from gsm import GSM
from utils import logsumexp
from distribution import Distribution

class Mixture(Distribution):
	"""
	A generic mixture class with an implementation of EM.

	B{References:}
		- M. Bishop (2006). I{Pattern Recognition and Machine Learning.}
		Springer Verlag.

	@type components: list
	@ivar components: mixture components

	@type priors: array_like
	@ivar priors: prior component weights

	@type alpha: positive real
	@ivar alpha: parameter of Dirichlet prior over component weights
	"""
	def __init__(self):
		self.components = []
		self.priors = None

		# parameter of regularizing Dirichlet prior
		self.alpha = 1.001



	def add_component(self, component):
		"""
		Add a component to the mixture distribution. This resets the parameters
		of the prior over the components.

		@type  component: Distribution
		@param component: a probabilistic model
		"""

		self.components.append(component)
		self.priors = ones(len(self)) / len(self)



	def __getitem__(self, key):
		"""
		Can be used to access components.

		@type  key: integer
		@param key: index of component
		"""

		return self.components[key]



	def __len__(self):
		"""
		Returns the number of components in the model.
		"""

		return len(self.components)



	def sample(self, num_samples):
		num_samples = multinomial(num_samples, self.priors)

		samples = []

		for i in range(len(self)):
			samples.append(self[i].sample(num_samples[i]))

		samples = concatenate(samples, 1)
		samples = samples[:, permutation(samples.shape[1])]

		return samples



	def train(self, data, weights=None, num_epochs=1):
		for epoch in range(num_epochs):
			#print '001'

			# compute posterior over components (E)
			post = exp(self.logposterior(data))

			print epoch, self.evaluate(data) / log(2)

			# incorporate conditional prior
			if weights is not None:
				post *= weights

			# adjust priors over components (M)
			self.priors = mean(post, 1)

			if self.alpha is not None:
				# regularization with Dirichlet prior
				self.priors += self.alpha - 1.
			self.priors /= sum(self.priors)

			# adjust remaining parameters (M)
			for i in range(len(self)):
				self[i].train(data, weights=post[i, :])



	def loglikelihood(self, data):
		# allocate memory
		logjoint = zeros([len(self), data.shape[1]])

		# compute joint density over components and data points
		for i in range(len(self)):
			logjoint[i, :] = self[i].loglikelihood(data) + log(self.priors[i])

		# marginalize
		return logsumexp(logjoint, 0)



	def logposterior(self, data):
		"""
		Computes the log-posterior distribution over components.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		# allocate memory
		logpost = zeros([len(self), data.shape[1]])

		# compute log-joint
		for i in range(len(self)):
			logpost[i, :] = self[i].loglikelihood(data) + log(self.priors[i])

		# normalize to get log-posterior
		logpost -= logsumexp(logpost, 0)

		return logpost



	def split(self, data):
		"""
		Randomly assigns data points to mixture components. The probability of a
		data point being assigned to a component is the posterior probability of
		the component given the data point.

		@type  data: array_like
		@param data: data stored in columns
		"""

		# compute posterior over components
		post = exp(self.logposterior(data))

		# sample indices from posterior
		uni = rand(data.shape[1])
		ind = zeros(uni.shape, 'int')
		cum = zeros(uni.shape)

		for k in range(1, len(self)):
			cum += post[k - 1, :]
			ind[uni > cum] = k

		# split data
		batches = []
		for k in range(len(self)):
			batches.append(data[:, ind == k])

		return batches
