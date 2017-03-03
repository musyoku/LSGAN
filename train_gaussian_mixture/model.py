# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear
from sequential.functions import Activation

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# specify discriminator
discriminator_sequence_filename = args.model_dir + "/discriminator.json"

if os.path.isfile(discriminator_sequence_filename):
	print "loading", discriminator_sequence_filename
	with open(discriminator_sequence_filename, "r") as f:
		try:
			discriminator_params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(discriminator_sequence_filename))
else:
	config = DiscriminatorParams()
	config.a = 0
	config.b = 1
	config.c = 1
	config.weight_std = 0.001
	config.weight_initializer = "Normal"
	config.use_weightnorm = False
	config.nonlinearity = "leaky_relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 1
	config.weight_decay = 0

	discriminator = Sequential()
	discriminator.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(128))
	discriminator.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(128))
	discriminator.add(Linear(None, 1, use_weightnorm=config.use_weightnorm))

	discriminator_params = {
		"config": config.to_dict(),
		"model": discriminator.to_dict(),
	}

	with open(discriminator_sequence_filename, "w") as f:
		json.dump(discriminator_params, f, indent=4, sort_keys=True, separators=(',', ': '))

# specify generator
generator_sequence_filename = args.model_dir + "/generator.json"

if os.path.isfile(generator_sequence_filename):
	print "loading", generator_sequence_filename
	with open(generator_sequence_filename, "r") as f:
		try:
			generator_params = json.load(f)
		except:
			raise Exception("could not load {}".format(generator_sequence_filename))
else:
	config = GeneratorParams()
	config.ndim_input = 256
	config.ndim_output = 2
	config.num_mixture = args.num_mixture
	config.distribution_output = "universal"
	config.use_weightnorm = False
	config.weight_std = 0.02
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 1
	config.weight_decay = 0

	# generator
	generator = Sequential()
	generator.add(Linear(config.ndim_input, 128, use_weightnorm=config.use_weightnorm))
	# generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	# generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, config.ndim_output, use_weightnorm=config.use_weightnorm))

	generator_params = {
		"config": config.to_dict(),
		"model": generator.to_dict(),
	}

	with open(generator_sequence_filename, "w") as f:
		json.dump(generator_params, f, indent=4, sort_keys=True, separators=(',', ': '))

gan = GAN(discriminator_params, generator_params)
gan.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	gan.to_gpu()