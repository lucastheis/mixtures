"""
Mixture of Gaussian scale mixtures convenience class.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from gsm import GSM
from utils import logsumexp
from mixture import Mixture

class MoGSM(Mixture):
	"""
	Mixture of Gaussian scale mixtures convenience class.
	"""

	def __init__(self, dim, num_components, num_scales):
		Mixture.__init__(self)

		# initialize components
		for i in range(num_components):
			self.add_component(GSM(dim, num_scales))
