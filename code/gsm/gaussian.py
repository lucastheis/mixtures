from numpy import cov, eye, mean, zeros, sum, multiply, pi, log, dot
from numpy.linalg import inv, det, slogdet

class Gaussian:
	def __init__(self, dim):
		self.dim = dim
		self.precision = eye(dim)
		self.mean = zeros([dim, 1])



	def train(self, data):
		self.precision = inv(cov(data))
		self.mean = mean(data, 1).reshape([dim, 1])



	def loglikelihood(self, data):
		data = data - self.mean

		_, logdet = slogdet(self.precision)

		tmp1 = (logdet - log(2. * pi) * self.dim) / 2.
		tmp2 = sum(multiply(data, dot(self.precision, data)), 0) / 2.

		return tmp1 - tmp2
