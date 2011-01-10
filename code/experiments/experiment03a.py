"""
Test Gaussianization and inverse Gaussianization.
"""

import sys

sys.path.append('./code')

from gsm import GSM
from matplotlib.pyplot import scatter, axis, figure, draw
from time import time

def main():
	gsm = GSM(2, 2)

	gsm.scales[0] = 10.
	gsm.scales[1] = 0.1

	samples = gsm.sample(2000)

	scatter(samples[0, :], samples[1, :], linewidth=0, alpha=0.5)
	axis('equal')
	draw()

	samples = gsm.gaussianize(samples)

	figure()
	scatter(samples[0, :], samples[1, :], linewidth=0, alpha=0.5)
	axis('equal')
	draw()

	samples = gsm.invgaussianize(samples)

	figure()
	scatter(samples[0, :], samples[1, :], linewidth=0, alpha=0.5)
	axis('equal')

	raw_input()



if __name__ == '__main__':
	sys.exit(main())
