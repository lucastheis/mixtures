"""
Provides an interface which should be implemented by all probabilistic models.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import mean

class Distribution:
	"""
	Provides an interface and common functionality for probabilistic models.
	"""

	def __init__(self):
		raise Exception(str(self.__class__) + ' is an abstract class.')



	def sample(self, num_samples=1):
		"""
		Generate samples from the model.

		@type  num_samples: integer
		@param num_samples: the number of samples to generate
		"""

		raise Exception('Abstract method \'sample\' not implemented in '
		    + str(self.__class__))



	def train(self, data, weights=None):
		"""
		Adapt the parameters of the model to the given set of data points.

		@type  data: array_like
		@param data: data stored in columns

		@type  weights: array_like
		@param weights: an optional weight for every data point
		"""

		raise Exception('Abstract method \'train\' not implemented in '
		    + str(self.__class__))



	def loglikelihood(self, data):
		"""
		Compute the log-likelihood of the model given the data.

		@type  data: array_like
		@param data: data stored in columns
		"""
		
		raise Exception('Abstract method \'loglikelihood\' not implemented in '
		    + str(self.__class__))



	def evaluate(self, data):
		"""
		Return average negative log-likelihood per dimension in nats.

		@type  data: array_like
		@param data: data stored in columns
		"""

		return -mean(self.loglikelihood(data)) / data.shape[0]
