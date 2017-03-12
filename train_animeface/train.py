import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import discriminator_params, generator_params, gan
from args import args
from dataset import load_rgb_images
from plot import plot

def get_learning_rate_for_epoch(epoch):
	if epoch < 50:
		return 0.0001
	if epoch < 100:
		return 0.00005
	if epoch < 150:
		return 0.00001
	if epoch < 200:
		return 0.000005
	return 0.000001

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x_batch[j] = images[data_index]
	return x_batch

def main():
	images = load_rgb_images(args.image_dir)

	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# labels
	a = discriminator_config.a
	b = discriminator_config.b
	c = discriminator_config.c

	# settings
	max_epoch = 1000
	num_updates_per_epoch = 500
	batchsize_true = 128
	batchsize_fake = 128
	plot_interval = 5

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_d = 0
		sum_loss_g = 0

		for t in xrange(num_updates_per_epoch):
			# sample data
			images_true = sample_from_data(images, batchsize_true)
			images_fake = gan.generate_x(batchsize_true)
			images_fake.unchain_backward()

			d_true = gan.discriminate(images_true, return_activations=False)
			d_fake = gan.discriminate(images_fake, return_activations=False)

			loss_d = 0.5 * (F.sum((d_true - b) ** 2) + F.sum((d_fake - a) ** 2)) / batchsize_true
			sum_loss_d += float(loss_d.data)

			# update discriminator
			gan.backprop_discriminator(loss_d)

			# generator loss
			images_fake = gan.generate_x(batchsize_fake)
			d_fake = gan.discriminate(images_fake, return_activations=False)
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
			plot(filename="epoch_{}_time_{}min".format(epoch, progress.get_total_time()))

if __name__ == "__main__":
	main()