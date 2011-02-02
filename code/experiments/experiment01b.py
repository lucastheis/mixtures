"""
Train mixture of GSMs on 8x8 van Hateren patches and evaluate model.
"""

import sys

sys.path.append('./code')

from models import MoGSM, MoGaussian
from numpy import load, log
from tools import preprocess, Experiment

# number of scales
parameters = [[8], [10], [12], [14], [16]]

def main(argv):
	experiment = Experiment()

	params = parameters[int(argv[1])]

	# load and preprocess data
	data = load('./data/vanhateren8x8.npz')['data']
	data = preprocess(data)

	# train a mixture of Gaussian scale mixtures
	mixture = MoGSM(data.shape[0], params[0], 12)
	mixture.train(data, num_epochs=70)

	# evaluate model
	avglogloss = mixture.evaluate(data) / log(2)

	# store results
	experiment.results['mixture'] = mixture
	experiment.results['avglogloss'] = avglogloss
	experiment.save('results/experiment01/experiment01b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
