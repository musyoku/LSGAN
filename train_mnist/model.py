# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
image_width = 28
image_height = image_width
ndim_latent_code = 50

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
	config.nonlinearity = "elu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.1
	config.gradient_clipping = 1
	config.weight_decay = 0

	discriminator = Sequential()
	discriminator.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	# discriminator.add(gaussian_noise(std=0.5))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(500))
	discriminator.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(500))
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
	config.ndim_input = ndim_latent_code
	config.ndim_output = image_width * image_height
	config.distribution_output = "tanh"
	config.use_weightnorm = False
	config.weight_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.1
	config.gradient_clipping = 1
	config.weight_decay = 0

	# generator
	generator = Sequential()
	generator.add(Linear(config.ndim_input, 500, use_weightnorm=config.use_weightnorm))
	generator.add(Activation(config.nonlinearity))
	# generator.add(BatchNormalization(500))
	generator.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	generator.add(Activation(config.nonlinearity))
	# generator.add(BatchNormalization(500))
	generator.add(Linear(None, config.ndim_output, use_weightnorm=config.use_weightnorm))
	if config.distribution_output == "sigmoid":
		generator.add(Activation("sigmoid"))
	if config.distribution_output == "tanh":
		generator.add(Activation("tanh"))

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