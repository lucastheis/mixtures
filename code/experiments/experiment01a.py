import sys

sys.path.append('./code')

from models import MoGSM, RadialGaussianization
from numpy import load, log, exp, concatenate
from tools import contours, Experiment

def main(argv):
	experiment = Experiment()

	# load preprocessed data samples
	data = load('./data/vanhateren.npz')
	data = concatenate([data['train'][1:, :], data['test'][1:, :]], 1)

	# train mixture of Gaussian scale mixtures
	mixture = MoGSM(15, 5, 12)
	mixture.train(data[:, :50000], num_epochs=100)

	# split data
	batches = mixture.split(data)

	# Gaussianize data
	for k in range(len(mixture)):
		batches[k] = RadialGaussianization(mixture[k])(batches[k])

	# store results
	experiment.results['mixture'] = mixture
	experiment.results['batches'] = batches
	experiment.save('results/experiment01/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
