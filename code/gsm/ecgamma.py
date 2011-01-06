from numpy import ones, zeros, zeros_like, dot, multiply, sum, mean, cov, eye
from numpy import square, sqrt, exp, log, pi, squeeze, diag, round, sort
from numpy.random import rand, randn
from numpy.linalg import inv, det, eig
from scipy.special import gamma, gammainc, gammaincinv
from utils import logsumexp

class ECGamma:
	def __init__(self, dim, shape=None, scale=1):
		pass



	def sample(self, num_samples=1):
		pass




	def train(self, data, posterior):
		pass



	def gaussianize(self, data):
		pass



	def loglikelihood(self, data):
		pass



	def logposterior(self, data):
		pass



	def logjoint(self, data):
		pass
