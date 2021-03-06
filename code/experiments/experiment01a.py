"""
Train mixture of GSMs on 4x4 van Hateren patches and radially Gaussianize data.
"""

import sys

sys.path.append('./code')

from models import MoGSM, RadialGaussianization
from numpy import load
from tools import Experiment, preprocess

def main(argv):
	experiment = Experiment()

	# load and preprocess data samples
	data = load('./data/vanhateren4x4.npz')['data']
	data = preprocess(data)

	# train mixture of Gaussian scale mixtures
	mixture = MoGSM(data.shape[0], 8, 4)
	mixture.train(data, num_epochs=100)

	# split data
	batches = mixture.split(data)

	# Gaussianize data
	for k in range(len(mixture)):
		batches[k] = RadialGaussianization(mixture[k], symmetric=False)(batches[k])

	# store results
	experiment.results['mixture'] = mixture
	experiment.results['batches'] = batches
	experiment.save('results/experiment01/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
