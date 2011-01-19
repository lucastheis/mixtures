"""
A collection of utility functions.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import log, sum, exp, zeros, max, asarray, vectorize, inf, nan
from scipy.special import gammainc
from scipy.optimize import bisect

def gammaincinv(a, y, maxiter=100):
	"""
	A slower but more stable implementation of the inverse regularized
	incomplete Gamma function.
	"""

	y_min = 0.

	if y > 1:
		return nan

	# make sure range includes root
	while gammainc(a, gammaincinv.y_max) < y:
		y_min = gammaincinv.y_max
		gammaincinv.y_max += 1.

	# find inverse with bisection method
	return bisect(
	    f=lambda x: gammainc(a, x) - y,
	    a=y_min,
	    b=gammaincinv.y_max,
	    maxiter=maxiter,
	    xtol=1e-16,
	    disp=True)

gammaincinv = vectorize(gammaincinv)
gammaincinv.y_max = 1



def logsumexp(x, ax=None):
	"""
	Computes the log of the sum of the exp of the entries in x in a numerically
	stable way.

	@type  x: array_like
	@param x: a list or a matrix of numbers

	@type  ax: integer
	@param ax: axis along which the sum is applied

	@rtype:  matrix
	@return: a matrix containing the results
	"""

	if ax is not None:
		output_shape = list(x.shape)
		output_shape[ax] = 1

		x_max = zeros(output_shape)
		max(x, ax, out=x_max)

		res = zeros(output_shape)
		sum(exp(x - x_max), ax, out=res)
		return x_max + log(res)
	else:
		x_max = x.max()
		return x_max + log(exp(x - x_max).sum(ax))



def logmeanexp(x, ax=None):
	"""
	Computes the log of the mean of the exp of the entries in x in a numerically
	stable way. Uses logsumexp.

	@type  x: array_like
	@param x: a list or a matrix of numbers

	@type  ax: integer
	@param ax: axis along which the values are averaged

	@rtype:  matrix
	@return: a matrix containing the results
	"""

	x = asarray(x)

	if ax is None:
		n = x.size
	else:
		n = x.shape[ax]

	return logsumexp(x, ax) - log(n)
