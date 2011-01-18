import sys

sys.path.append('./code')

from models import MoGSM
from numpy import load, log
from tools import preprocess, Experiment

def main(argv):
	experiment = Experiment()

	# load and preprocess data
	data = load('./data/patches16x16.npz')['data']
	data = preprocess(data)

	# train a mixture of Gaussian scale mixtures
	mixture = MoGSM(data.shape[0], 5, 12)
	mixture.train(data, num_epochs=100)

	# evaluate model
	avglogloss = mixture.evaluate(data) / log(2)

	# store results
	experiment.results['mixture'] = mixture
	experiment.results['avglogloss'] = avglogloss
	experiment.save('results/experiment01/experiment01b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
