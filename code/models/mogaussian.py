from gaussian import Gaussian
from utils import logsumexp
from mixture import Mixture

class MoGaussian(Mixture):
	def __init__(self, dim, num_components):
		Mixture.__init__(self)

		# initialize components
		for i in range(num_components):
			self.add_component(Gaussian(dim))
