# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj
	
class Params():
	def __init__(self, dict=None):
		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			if hasattr(value, "to_dict"):
				dict[attr] = value.to_dict()
			else:
				dict[attr] = value
		return dict

	def dump(self):
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

class DiscriminatorParams(Params):
	def __init__(self):
		self.a = 0
		self.b = 1
		self.c = 1
		self.weight_std = 0.001
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "leaky_relu"
		self.optimizer = "adam"
		self.learning_rate = 0.0001
		self.momentum = 0.5
		self.gradient_clipping = 1
		self.weight_decay = 0

class GeneratorParams(Params):
	def __init__(self):
		self.ndim_input = 256
		self.ndim_output = 2
		self.num_mixture = 8
		self.distribution_output = "universal"	# universal, sigmoid or tanh
		self.weight_std = 0.02
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "adam"
		self.learning_rate = 0.0001
		self.momentum = 0.5
		self.gradient_clipping = 1
		self.weight_decay = 0

class GAN():
	def __init__(self, params_discriminator, params_generator):
		self.params_discriminator = copy.deepcopy(params_discriminator)
		self.config_discriminator = to_object(params_discriminator["config"])

		self.params_generator = copy.deepcopy(params_generator)
		self.config_generator = to_object(params_generator["config"])

		self.build_discriminator()
		self.build_generator()
		self._gpu = False

	def build_discriminator(self):
		config = self.config_discriminator
		self.discriminator = sequential.chain.Chain(weight_initializer=config.weight_initializer, weight_std=config.weight_std)
		self.discriminator.add_sequence(sequential.from_dict(self.params_discriminator["model"]))
		self.discriminator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def build_generator(self):
		config = self.config_generator
		self.generator = sequential.chain.Chain(weight_initializer=config.weight_initializer, weight_std=config.weight_std)
		self.generator.add_sequence(sequential.from_dict(self.params_generator["model"]))
		self.generator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def update_learning_rate(self, lr):
		self.discriminator.update_learning_rate(lr)
		self.generator.update_learning_rate(lr)

	def to_gpu(self):
		self.discriminator.to_gpu()
		self.generator.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def sample_z(self, batchsize=1, gaussian=False):
		config = self.config_generator
		ndim_z = config.ndim_input
		if gaussian:
			# gaussian
			z_batch = np.random.normal(0, 1, (batchsize, ndim_z)).astype(np.float32)
		else:
			# uniform
			z_batch = np.random.uniform(-1, 1, (batchsize, ndim_z)).astype(np.float32)
		return z_batch

	def generate_x(self, batchsize=1, test=False, as_numpy=False, from_gaussian=False):
		return self.generate_x_from_z(self.sample_z(batchsize, gaussian=from_gaussian), test=test, as_numpy=as_numpy)

	def generate_x_from_z(self, z_batch, test=False, as_numpy=False):
		z_batch = self.to_variable(z_batch)
		x_batch, _ = self.generator(z_batch, test=test, return_activations=True)
		if as_numpy:
			return self.to_numpy(x_batch)
		return x_batch

	def discriminate(self, x_batch, test=False, return_activations=False):
		x_batch = self.to_variable(x_batch)
		out = self.discriminator(x_batch, test=test, return_activations=return_activations)
		if return_activations:
			return out[0], out[1]
		return out

	def backprop_discriminator(self, loss):
		self.discriminator.backprop(loss)

	def backprop_generator(self, loss):
		self.generator.backprop(loss)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.generator.load(dir + "/generator.hdf5")
		self.discriminator.load(dir + "/discriminator.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.generator.save(dir + "/generator.hdf5")
		self.discriminator.save(dir + "/discriminator.hdf5")