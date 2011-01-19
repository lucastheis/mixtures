"""
Tools for extracting and displaying image patches.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'
__version__ = '1.0.0'

from numpy import array, asarray, floor, ceil, sqrt, zeros
from numpy.random import uniform
import matplotlib.pyplot as mplt

def sample(img, patch_size, num_samples):
	"""
	Generates a random sample of image patches from an image.

	@type  img: array_like
	@param img: a grayscale image
	
	@type  patch_size: tuple
	@param patch_size: height and width of patches

	@type  num_samples: integer
	@param num_samples: number of samples

	@rtype: ndarray
	@return: patches stored in an MxNxK array
	"""

	# uniformly sample patch locations
	xpos = floor(uniform(0, img.shape[0] - patch_size[0] + 1, num_samples))
	ypos = floor(uniform(0, img.shape[1] - patch_size[1] + 1, num_samples))

	# collect sample patches
	samples = []
	for i in range(num_samples):
		samples.append(img[xpos[i]:xpos[i] + patch_size[0], ypos[i]:ypos[i] + patch_size[1]])

	return array(samples)



def show(samples, num_rows=None, num_cols=None, num_patches=None, line_width=1, margin=20):
	"""
	Displays a sample of image patches.

	The patches should be stored in a KxMxN array, where K is the number
	samples, M is the number of rows and N is the number of columns in each
	patch.

	@type  samples: array_like
	@param samples: patches stored in an MxNxK array

	@type  num_rows: integer
	@param num_rows: number of rows of patches

	@type  num_cols: integer
	@param num_cols: number of columns of patches

	@type  num_patches: integer
	@param num_patches: only display the first num_patches patches

	@type  line_width: integer
	@param line_width: distance between patches

	@type  margin: integer
	@param margin: margin of figure
	"""

	# process and check parameters
	samples = asarray(samples)

	if not num_patches:
		num_patches = samples.shape[0]
	if not num_rows and not num_cols:
		num_cols = ceil(sqrt(num_patches))
		num_rows = ceil(num_patches / num_cols)
	elif not num_rows:
		num_rows = ceil(num_patches / num_cols)
	elif not num_cols:
		num_cols = ceil(num_patches / num_rows)

	num_patches = int(min(min(num_patches, samples.shape[0]), num_rows * num_cols))
	num_rows = int(num_rows)
	num_cols = int(num_cols)

	patch_size = samples.shape[1:3]

	# normalize patches
	smin = float(samples.min())
	smax = float(samples.max())
	samples = (samples - smin) / (smax - smin)

	# allocate memory
	if len(samples.shape) > 3:
		patchwork = zeros((
				num_rows * patch_size[0] + (num_rows + 1) * line_width,
				num_cols * patch_size[1] + (num_cols + 1) * line_width, 3))
	else:
		patchwork = zeros((
				num_rows * patch_size[0] + (num_rows + 1) * line_width,
				num_cols * patch_size[1] + (num_cols + 1) * line_width))

	# stitch patches together
	for i in range(num_patches):
		r = i / num_cols
		c = i % num_cols

		r_off = r * patch_size[0] + (r + 1) * line_width
		c_off = c * patch_size[1] + (c + 1) * line_width

		patchwork[r_off:r_off + patch_size[0], c_off:c_off + patch_size[1], ...] = samples[i]

	# display patches
	h = mplt.imshow(patchwork,
	    cmap='gray',
	    interpolation='nearest',
	    aspect='equal')

	xmargin = float(margin) / (patchwork.shape[1] + 2 * margin + 1)
	ymargin = float(margin) / (patchwork.shape[0] + 2 * margin + 1)
	xwidth = 1 - 2 * xmargin
	ywidth = 1 - 2 * ymargin

	# make sure that 1 image pixel is represented by 1 pixel of the screen
	dpi = h.figure.get_dpi()
	h.figure.set_figwidth(patchwork.shape[0] / (ywidth * dpi))
	h.figure.set_figheight(patchwork.shape[1] / (xwidth * dpi))
	h.figure.canvas.resize(patchwork.shape[1] + 2 * margin + 1, patchwork.shape[0] + 2 * margin + 1)

	h.axes.set_position([xmargin, ymargin, xwidth, ywidth])
	h.axes.set_xlim(-1, patchwork.shape[1])
	h.axes.set_ylim(patchwork.shape[0], -1)

	mplt.axis('off')
	mplt.draw()

	return h
