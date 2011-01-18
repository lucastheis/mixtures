import sys

sys.path.append('./code')

from models import MoGSM
from numpy import load, log, exp

def main(argv):
	# load preprocessed data samples
	data = load('./data/vanhateren.npz')

	mixture = MoGSM(15, 5, 12)
	mixture.train(data['train'][1:, :], num_epochs=100)

	print mixture.evaluate(data['train'][1:, :]) / log(2),
	print mixture.evaluate(data['test'][1:, :]) / log(2)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
