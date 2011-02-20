"""
Train mixture of GSMs on 8x8 van Hateren patches and evaluate model.
"""

import sys

sys.path.append('./code')

from models import MoGSM, MoGaussian
from numpy import load, log, mean
from tools import Experiment, preprocess

def main(argv):
	experiment = Experiment()

	# load and preprocess data
	data = load('./data/vanhateren8x8.npz')['data']
#	data = preprocess(data)
	data = log(data + 1.)
	data -= mean(data)

	# train a mixture of Gaussian scale mixtures
	mixture = MoGSM(data.shape[0], 8, 4)
	mixture.train(data[:, :100000], num_epochs=100)

	# compute training error
	avglogloss = mixture.evaluate(data[:, 100000:])

	# store results
	experiment.results['mixture'] = mixture
	experiment.results['avglogloss'] = avglogloss
	experiment.save('results/experiment01/experiment01b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
