"""
This module simplifies modeling of transformed data.
"""

class Transform:
	"""
	Handles modeling of transformed data.
	"""

	def __init__(self, model, function, inverse=None, logjacobian=None):
		self.function = function
		self.inverse = inverse
		self.logjacobian = logjacobian
		self.model = model



	def sample(self, num_samples=1):
		if self.function is None:
			raise RuntimeError("Inverse function required for sampling.")

		return self.inverse(self.model.sample(num_samples))



	def train(self, data):
		self.model.train(self.function(data))



	def loglikelihood(self, data):
		if self.function is None:
			raise RuntimeError("Jacobian determinant required for evaluation.")

		return \
			self.model.loglikelihood(self.function(data)) + \
			self.logjacobian(data)
