import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import sampler
from progress import Progress
from model import discriminator_params, generator_params, gan
from args import args
from plot import plot_kde, plot_scatter

def plot_samples(epoch, progress):
	samples_fale = gan.generate_x(10000, from_gaussian=True)
	samples_fale.unchain_backward()
	samples_fale = gan.to_numpy(samples_fale)
	try:
		plot_scatter(samples_fale, dir=args.plot_dir, filename="scatter_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
		plot_kde(samples_fale, dir=args.plot_dir, filename="kde_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
	except:
		pass

def main():
	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# labels
	a = discriminator_config.a
	b = discriminator_config.b
	c = discriminator_config.c

	# settings
	max_epoch = 200
	num_updates_per_epoch = 500
	plot_interval = 5
	batchsize_true = 100
	batchsize_fake = batchsize_true
	scale = 2.0

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	progress = Progress()
	plot_samples(0, progress)
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_d = 0
		sum_loss_g = 0

		for t in xrange(num_updates_per_epoch):
			# sample from data distribution
			samples_true = sampler.gaussian_mixture_circle(batchsize_true, generator_config.num_mixture, scale=scale, std=0.2)
			# sample from generator
			samples_fale = gan.generate_x(batchsize_true, from_gaussian=True)
			samples_fale.unchain_backward()

			d_true = gan.discriminate(samples_true / scale, return_activations=False)
			d_fake = gan.discriminate(samples_fale / scale, return_activations=False)

			loss_d = 0.5 * (F.sum((d_true - b) ** 2) + F.sum((d_fake - a) ** 2)) / batchsize_true
			sum_loss_d += float(loss_d.data)

			# update discriminator
			gan.backprop_discriminator(loss_d)

			# generator loss
			samples_fale = gan.generate_x(batchsize_fake, from_gaussian=True)
			d_fake = gan.discriminate(samples_fale / scale, return_activations=False)
			loss_g = 0.5 * (F.sum((d_fake - c) ** 2)) / batchsize_fake
			sum_loss_g += float(loss_g.data)

			# update generator
			gan.backprop_generator(loss_g)
			
			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		gan.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss_d": sum_loss_d / num_updates_per_epoch,
			"loss_g": sum_loss_g / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot_samples(epoch, progress)

if __name__ == "__main__":
	main()
