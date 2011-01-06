from numpy import multiply, dot, sum, mean, cov, sqrt, log, exp, pi, argsort
from numpy import ones, zeros, zeros_like, eye, round, squeeze, concatenate
from numpy.random import multinomial
from numpy.linalg import det, inv, eig
from gsm import GSM
from utils import logsumexp

class Mixture:
	def __init__(self):
		self.components = []
		self.priors = None

		# parameter of regularizing Dirichlet prior
		self.alpha = 1.001



	def add_component(self, component):
		self.components.append(component)
		self.priors = ones(len(self)) / len(self)



	def __getitem__(self, key):
		return self.components[key]



	def __len__(self):
		return len(self.components)



	def sample(self, num_samples):
		num_samples = multinomial(num_samples, self.priors)

		samples = []

		for i in range(len(self)):
			samples.append(self[i].sample(num_samples[i]))

		return concatenate(samples, 1)



	def train(self, data, data_valid, num_epochs=100):
		# allocate memory
		logpost = zeros([len(self), data.shape[1]])

		for epoch in range(num_epochs):
			# compute posterior over components and scales (E)
			for i in range(len(self)):
				logpost[i, :] = self[i].loglikelihood(data) + log(self.priors[i])
			logpost -= logsumexp(logpost, 0)

			print round(self.avglogloss(data) / log(2) / self[0].dim, 4),
			print round(self.avglogloss(data_valid) / log(2) / self[0].dim, 4)

			# adjust priors over components (M)
			self.priors = mean(exp(logpost), 1) + (self.alpha - 1.)
			self.priors /= sum(self.priors)

			# adjust remaining parameters (M)
			for i in range(len(self)):
				self[i].train(data, exp(logpost[i, :]))



	def loglikelihood(self, data):
		# allocate memory
		logjoint = zeros([len(self), data.shape[1]])

		# compute joint density over components and data points
		for i in range(len(self)):
			logjoint[i, :] = self[i].loglikelihood(data) + log(self.priors[i])

		# marginalize
		return logsumexp(logjoint, 0)



	def avglogloss(self, data):
		return -mean(self.loglikelihood(data))
