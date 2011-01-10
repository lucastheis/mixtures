"""
Test Gaussianization with Transform class.
"""

import sys

sys.path.append('./code')

from gsm import Transform, GSM, Gaussian
from numpy import log, abs



def main():
	dim = 15

	gss = Gaussian(dim)
	gsm = GSM(dim, 2)
	gtr = Transform(gss, gsm.gaussianize, gsm.invgaussianize, gsm.logjacobian)

	gsm.scales[0] = 0.1
	gsm.scales[1] = 10.

	samples = gsm.sample(10000)

	print (gsm.loglikelihood(samples).mean() - gtr.loglikelihood(samples).mean()) / dim

	return 0



if __name__ == '__main__':
	sys.exit(main())
