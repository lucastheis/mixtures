from numpy import multiply, dot, sum, mean, cov, sqrt, log, exp, pi, argsort
from numpy import ones, zeros, zeros_like, eye, round, squeeze, concatenate
from numpy.random import multinomial
from numpy.linalg import det, inv, eig
from gsm import GSM
from utils import logsumexp

class MoGSM:
	def __init__(self, dim, num_components, num_scales):
		self.dim = dim
		self.num_components = num_components
		self.num_scales = num_scales

		# initial prior over components
		self.priors = ones(num_components) / num_components

		# initialize components
		self.components = []
		for i in range(num_components):
			self.components.append(GSM(dim, num_scales))



	def __getitem__(self, key):
		return self.components[key]



	def __len__(self):
		return self.num_components



	def sample(self, num_samples):
		num_samples = multinomial(num_samples, self.priors)

		samples = []

		for i in range(self.num_components):
			samples.append(self[i].sample(num_samples[i]))

		return concatenate(samples, 1)



	def train(self, data, data_valid, num_epochs=100):
		# allocate memory
		logpost = zeros([self.num_components, self.num_scales, data.shape[1]])

		for epoch in range(num_epochs):
			# compute posterior over components and scales (E)
			for i in range(self.num_components):
				logpost[i, :, :] = self[i].logjoint(data) + log(self.priors[i])
			logpost -= logsumexp(logsumexp(logpost, 0), 1)

			# adjust priors over components (M)
			self.priors = mean(sum(exp(logpost), 1), 1)

			self.priors += 0.001
			self.priors /= sum(self.priors)

			# adjust remaining parameters (M)
			for i in range(self.num_components):
				self[i].train(data, exp(logpost[i, :, :]))


			for i in range(self.num_components):
				indices = argsort(self[i].scales)



	def logloss(self, data):
			logpost = zeros([self.num_components, self.num_scales, data.shape[1]])
			for i in range(self.num_components):
				logpost[i, :, :] = self[i].logjoint(data) + log(self.priors[i])
			return -mean(logsumexp(logsumexp(logpost, 0), 1)) / log(2) / self.dim
