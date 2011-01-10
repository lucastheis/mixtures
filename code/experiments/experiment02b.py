import sys

sys.path.append('./code')

from gsm import MoGSM, GSM
from numpy import load
from time import time
from matplotlib.pyplot import scatter, figure, axis, draw
from numpy.random import randn, permutation
from tools import preprocess



def main(argv):
	data = load('./data/vanhateren.npz')
	data_train = data['train'][1:, :]
	data_valid = data['test'][1:, :]

	mogsm = MoGSM(15, 5, 8)
	mogsm.train(data_train, data_valid, 100)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
