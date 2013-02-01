"""
A generic mixture class with an implementation of EM.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import multiply, dot, sum, mean, cov, sqrt, log, exp, pi, argsort
from numpy import ones, zeros, zeros_like, eye, round, squeeze, concatenate
from numpy import asarray
from numpy.random import multinomial, rand, permutation
from numpy.linalg import det, inv, eig
from gsm import GSM
from utils import logsumexp
from distribution import Distribution
from tools.parallel import map
from tools import shmarray

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
		self.initialized = False

		# parameter of regularizing Dirichlet prior
		self.alpha = None#2.



	def add_component(self, component):
		"""
		Add a component to the mixture distribution. This resets the parameters
		of the prior over the components.

		@type  component: L{Distribution}
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



	def train(self, data, weights=None, num_epochs=100, threshold=1e-5):
		"""
		Adapt the parameters of the model using expectation maximization (EM).

		@type  data: array_like
		@param data: data stored in columns

		@type  weights: array_like
		@param weights: an optional weight for every data point

		@type  num_epochs: integer
		@param num_epochs: maximum number of training epochs

		@type  threshold: float
		@param threshold: training stops if performance gain is below threshold
		"""

		if not self.initialized:
			# initialize components
			def initialize_(i):
				self.components[i].initialize(data)
				return self.components[i]
			self.components = map(initialize_, range(len(self)), max_processes=1)
			self.initialized = True

		# current performance
		value = self.evaluate(data)

		if Distribution.VERBOSITY >= 2:
			print 'epoch 0\t', value

		for epoch in range(num_epochs):
			# compute posterior over components (E)
			post = exp(self.logposterior(data))
			post /= sum(post, 0)

			# incorporate conditional prior
			if weights is not None:
				post *= weights

			# adjust priors over components (M)
			self.priors = sum(post, 1)

			if self.alpha is not None:
				# regularization with Dirichlet prior
				self.priors += self.alpha - 1.
			self.priors /= sum(self.priors)

			# adjust components (M)
			def train_(i):
				self.components[i].train(data, weights=post[i, :])
				return self.components[i]
			self.components = map(train_, range(len(self)))

			# check for convergence
			new_value = self.evaluate(data)

			if Distribution.VERBOSITY >= 2:
				print 'epoch ', epoch + 1, '\t', new_value

			if value - new_value < threshold:
				if Distribution.VERBOSITY >= 1:
					print 'training converged...'
				return
			value = new_value

		if Distribution.VERBOSITY >= 1:
			print 'training finished...'



	def loglikelihood(self, data):
		# allocate memory
		logjoint = shmarray.zeros([len(self), data.shape[1]])

		# compute joint density over components and data points
		def loglikelihood_(i):
			logjoint[i, :] = self[i].loglikelihood(data) + log(self.priors[i])
		map(loglikelihood_, range(len(self)))

		# marginalize
		return asarray(logsumexp(logjoint, 0)).flatten()



	def logposterior(self, data):
		"""
		Computes the log-posterior distribution over components.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		# allocate memory
		logpost = shmarray.zeros([len(self), data.shape[1]])

		# compute log-joint
		def logposterior_(i):
			logpost[i, :] = self[i].loglikelihood(data) + log(self.priors[i])
		map(logposterior_, range(len(self)))

		# normalize to get log-posterior
		logpost -= logsumexp(logpost, 0)

		return asarray(logpost)



	def split(self, data):
		"""
		Randomly assigns data points to mixture components.
		
		The probability of a data point being assigned to a component is the
		posterior probability of the component given the data point.

		@type  data: array_like
		@param data: data stored in columns

		@rtype: C{list}
		@return: list of arrays containing the data
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



	def __getstate__(self):
		"""
		Called upon pickling.

		@rtype: dictionary
		@return: the member variables of an MoCGSM instance
		"""

		# turn shared memory arrays into regular arrays
		for key, value in self.__dict__.iteritems():
			if type(value) is shmarray.shmarray:
				self.__dict__[key] = asarray(value)
		return self.__dict__



	def __setstate__(self, state):
		"""
		Called upon unpickling.

		@type  state: dictionary
		@param state: the member variables of an MoCGSM instance
		"""

		self.__dict__ = state
