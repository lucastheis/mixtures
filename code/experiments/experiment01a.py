"""
Check if training can recover parameters.
"""

import sys

sys.path.append('./code')

from gsm import MoGSM, GSM
from numpy import max, min
from time import time
from matplotlib.pyplot import scatter, figure, axis, draw
from numpy.random import randn, permutation



def main(argv):
	mogsm = MoGSM(2, 2, 5)

	# sample from true model
	samples = mogsm.sample(5000)

	scatter(samples[0, :], samples[1, :], linewidth=0, alpha=0.5)
	axis('equal')

	# relearn parameters
	mogsm = MoGSM(2, 2, 5)
	mogsm.train(samples, samples, 20)

	# sample from recovered model
	samples = mogsm.sample(5000)

	figure()
	scatter(samples[0, :], samples[1, :], linewidth=0, alpha=0.5)
	axis('equal')

	raw_input()

	return 0


if __name__ == '__main__':
	sys.exit(main(sys.argv))
