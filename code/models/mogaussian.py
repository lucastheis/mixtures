"""
Mixture of Gaussians convenience class.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from gaussian import Gaussian
from utils import logsumexp
from mixture import Mixture

class MoGaussian(Mixture):
	"""
	Mixture of Gaussians convenience class.
	"""

	def __init__(self, dim, num_components):
		Mixture.__init__(self)

		# initialize components
		for i in range(num_components):
			self.add_component(Gaussian(dim))
