from numpy import multiply, dot, sum, mean, cov, sqrt, log, exp, pi, argsort
from numpy import ones, zeros, zeros_like, eye, round, squeeze, concatenate
from numpy.random import multinomial
from numpy.linalg import det, inv, eig
from gsm import GSM
from utils import logsumexp
from mixture import Mixture

class MoGSM(Mixture):
	def __init__(self, dim, num_components, num_scales):
		Mixture.__init__(self)

		# initialize components
		for i in range(num_components):
			self.add_component(GSM(dim, num_scales))
