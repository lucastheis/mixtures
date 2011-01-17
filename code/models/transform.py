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

	def __init__(self, model, function, inverse=None, logjacobian=None):
		self.function = function
		self.inverse = inverse
		self.logjacobian = logjacobian
		self.model = model



	def sample(self, num_samples=1, *args, **kwargs):
		if self.function is None:
			raise RuntimeError("Inverse function required for sampling.")

		return self.inverse(self.model.sample(num_samples, *args, **kwargs))



	def train(self, data, *args, **kwargs):
		self.model.train(self.function(data), *args, **kwargs)



	def loglikelihood(self, data):
		if self.function is None:
			raise RuntimeError("Jacobian determinant required for evaluation.")

		return \
			self.model.loglikelihood(self.function(data)) + \
			self.logjacobian(data)
