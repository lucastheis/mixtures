from numpy import log, transpose, mean, dot, diag, sqrt, cov
from numpy.random import permutation
from numpy.linalg import eig

def preprocess(data):
	# log-transform
	data = log(data + 1.)

	# center
	data = data - mean(data, 1).reshape(-1, 1)

	# shuffle
	data = data[:, permutation(data.shape[1])]

	# find eigenvectors
	eigvals, eigvecs = eig(cov(data))

	# eliminate eigenvectors whose eigenvalues are zero
	eigvecs = eigvecs[:, eigvals > 0]
	eigvals = eigvals[eigvals > 0]

	# symmetric whitening matrix
	whitening_matrix = dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))

	# whiten data
	return dot(whitening_matrix, data)
