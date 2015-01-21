#!/usr/bin/env python

"""
Manage and display experimental results.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'
__version__ = '0.3.2'

import sys
import os
import numpy
import scipy

sys.path.append('./code')

from pickle import Unpickler, dump
from subprocess import Popen, PIPE
from os import path
from warnings import warn
from time import time, strftime, localtime
from numpy import random
from numpy.random import rand
from distutils.version import StrictVersion

class Experiment:
	"""
	@type time: float
	@ivar time: time at initialization of experiment

	@type duration: float
	@ivar duration: time in seconds between initialization and saving

	@type platform: string
	@ivar platform: information about operating system

	@type processors: string
	@ivar processors: some information about the processors

	@type environ: string
	@ivar environ: environment variables at point of initialization

	@type comment: string
	@ivar comment: a comment describing the experiment

	@type results: dictionary
	@ivar results: container to store experimental results

	@type commit: string
	@ivar commit: git commit hash

	@type modified: boolean
	@ivar modified: indicates uncommited changes

	@type filename: string
	@ivar filename: path to stored results

	@type seed: int
	@ivar seed: random seed used through the experiment

	@type versions: dictionary
	@ivar versions: versions of Python, numpy and scipy
	"""

	def __str__(self):
		"""
		Summarize information about the experiment.

		@rtype: string
		@return: summary of the experiment
		"""

		strl = []

		# date and duration of experiment
		strl.append(strftime('date \t\t %a, %d %b %Y %H:%M:%S', localtime(self.time)))
		strl.append('duration \t ' + str(int(self.duration)) + 's')

		# commit hash
		if self.commit:
			if self.modified:
				strl.append('commit \t\t ' + self.commit + ' (modified)')
			else:
				strl.append('commit \t\t ' + self.commit)

		# results
		strl.append('results \t {' + ', '.join(map(str, self.results.keys())) + '}')

		# comment
		if self.comment:
			strl.append('\n' + self.comment)

		return '\n'.join(strl)



	def __init__(self, filename="", comment="", seed=None):
		"""
		If the filename is given and points to an existing experiment, load it.
		Otherwise store the current timestamp and try to get commit information
		from the repository in the current directory.

		@type  filename: string
		@param filename: path to where the experiment will be stored
		
		@type comment: string
		@param comment: a comment describing the experiment

		@type  seed: integer
		@param seed: random seed used in the experiment
		"""

		self.time = time()
		self.comment = comment
		self.filename = filename
		self.results = {}
		self.seed = seed
		self.platform = ''
		self.processors = ''
		self.environ = ''
		self.duration = 0
		self.versions = {}

		if self.seed is None:
			self.seed = int((time() + 1e6 * rand()) * 1e3) % 4294967295

		# set random seed
		random.seed(self.seed)
		numpy.random.seed(self.seed)

		if path.isfile(self.filename):
			# load given experiment
			self.load()
		else:
			# get OS information
			self.platform = sys.platform

			# arguments to the program
			self.argv = sys.argv

			# environment variables
			self.environ = os.environ

			# store some information about the processor(s)
			if self.platform == 'linux2':
				cmd = 'egrep "processor|model name|cpu MHz|cache size" /proc/cpuinfo'
				with os.popen(cmd) as handle:
					self.processors = handle.read()
			elif self.platform == 'darwin':
				cmd = 'system_profiler SPHardwareDataType | egrep "Processor|Cores|L2|Bus"'
				with os.popen(cmd) as handle:
					self.processors = handle.read()

			# version information
			self.versions['python'] = sys.version
			self.versions['numpy'] = numpy.__version__
			self.versions['scipy'] = scipy.__version__

			# store information about git repository
			if path.isdir('.git'):
				# get commit hash
				pr1 = Popen(['git', 'log', '-1'], stdout=PIPE)
				pr2 = Popen(['head', '-1'], stdin=pr1.stdout, stdout=PIPE)
				pr3 = Popen(['cut', '-d', ' ', '-f', '2'], stdin=pr2.stdout, stdout=PIPE)
				self.commit = pr3.communicate()[0][:-1]

				# check if project contains uncommitted changes
				pr1 = Popen(['git', 'status', '--porcelain'], stdout=PIPE)
				pr2 = Popen(['egrep', '^.M'], stdin=pr1.stdout, stdout=PIPE)
				self.modified = pr2.communicate()[0]

				if self.modified:
					warn("Uncommitted changes.")
			else:
				# no git repository
				self.commit = None
				self.modified = False



	def save(self, filename=None):
		"""
		Store results. If a filename is given, the default is overwritten.

		@type  filename: string
		@param filename: path to where the experiment will be stored
		"""

		self.duration = time() - self.time

		if filename is None:
			filename = self.filename

		# replace {0} and {1} by date and time
		tmp1 = strftime('%d%m%Y', localtime(time()))
		tmp2 = strftime('%H%M%S', localtime(time()))
		filename = filename.format(tmp1, tmp2)

		# make sure filename does not exist
		counter = 0
		pieces = path.splitext(filename)
		while path.exists(filename):
			counter += 1
			filename = pieces[0] + '.' + str(counter) + pieces[1]

		if counter:
			warn(''.join(pieces) + ' already exists. Saving to ' + filename + '.')

		# store experiment
		with open(filename, 'wb') as handle:
			dump({
				'version': __version__,
				'time': self.time,
				'seed': self.seed,
				'duration': self.duration,
				'environ': self.environ,
				'argv': self.argv,
				'processors': self.processors,
				'platform': self.platform,
				'comment': self.comment,
				'commit': self.commit,
				'modified': self.modified,
				'versions': self.versions,
				'results': self.results}, handle, 1)



	def load(self, filename=None):
		"""
		Loads experimental results from the specified file.

		@type  filename: string
		@param filename: path to where the experiment is stored
		"""

		if filename:
			self.filename = filename

		with open(self.filename, 'rb') as handle:
			res = load(handle)

			self.time = res['time']
			self.seed = res['seed']
			self.duration = res['duration']
			self.processors = res['processors']
			self.environ = res['environ']
			self.platform = res['platform']
			self.comment = res['comment']
			self.commit = res['commit']
			self.modified = res['modified']
			self.versions = res['versions']
			self.results = res['results']

			if StrictVersion(res['version']) >= '0.3.1':
				self.argv = res['argv']
			else:
				# argv unknown
				self.argv = None



	def __getitem__(self, key):
		return self.results[key]



	def __setitem__(self, key, value):
		self.results[key] = value



class XUnpickler(Unpickler):
	"""
	An extension of the Unpickler class which resolves some backwards
	compatibility issues of Numpy.
	"""

	def find_class(self, module, name):
		"""
		Helps Unpickler to find certain Numpy modules.
		"""

		try:
			numpy_version = StrictVersion(numpy.__version__)

			if numpy_version >= '1.5.0':
				if module == 'numpy.core.defmatrix':
					module = 'numpy.matrixlib.defmatrix'

		except ValueError:
			pass

		return Unpickler.find_class(self, module, name)



def load(file):
	return XUnpickler(file).load()



def main(argv):
	"""
	Load and display experiment information.
	"""

	if len(argv) < 2:
		print 'Usage:', argv[0], '<filename>'
		return 0

	# load experiment
	experiment = Experiment(sys.argv[1])

	if len(argv) > 2:
		# print arguments
		for arg in argv[2:]:
			try:
				print experiment[arg]
			except:
				print experiment[int(arg)]
		return 0

	# print summary of experiment
	print experiment

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
