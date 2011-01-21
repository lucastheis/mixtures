"""
Simplifies modeling of transformed data.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from distribution import Distribution

class Transform(Distribution):
	"""
	Simplifies modeling of transformed data.
	"""

	def __init__(self, function, inverse=None, logjacobian=None, model=None):
		self.function = function
		self.inverse = inverse
		self.logjacobian = logjacobian
		self.model = model



	def __call__(self, data):
		"""
		Applies the transformation to the given set of data points.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		return self.function(data)



	def sample(self, num_samples=1, *args, **kwargs):
		if self.function is None:
			raise RuntimeError("Inverse function required for sampling.")

		if self.model is None:
			raise RuntimeError("Model required for sampling.")

		return self.inverse(self.model.sample(num_samples, *args, **kwargs))



	def train(self, data, *args, **kwargs):
		if self.model is None:
			raise RuntimeError("Model required for training.")

		self.model.train(self.function(data), *args, **kwargs)



	def loglikelihood(self, data):
		if self.function is None:
			raise RuntimeError("Jacobian determinant required for evaluation.")

		if self.model is None:
			raise RuntimeError("Model required for evaluation.")

		return \
			self.model.loglikelihood(self.function(data)) + \
			self.logjacobian(data)
