import sys

sys.path.append('./code')

from models import MoGSM
from numpy import load, log, exp

def main(argv):
	data = load('./data/vanhateren.npz')

	mixture = MoGSM(15, 5, 12)
	mixture.train(data['train'][1:, :], num_epochs=100)

	print mixture.avglogloss(data['train'][1:, :]) / 15. / log(2)
	print mixture.avglogloss(data['test'][1:, :]) / 15. / log(2)

	posterior = exp(mixture.logposterior(data['test'][1:, :]))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
