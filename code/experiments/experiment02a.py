import sys

sys.path.append('./code')

from gsm import MoGSM, GSM
from numpy import load
from time import time
from matplotlib.pyplot import scatter, figure, axis, draw
from numpy.random import randn, permutation
from tools import preprocess



def main(argv):
	data = load('./data/patches16x16.npz')['data']
	data = preprocess(data)

	mogsm = MoGSM(256, 8, 8)
	mogsm.train(data, data, 50)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
