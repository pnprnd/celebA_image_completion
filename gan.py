import tensorflow as tf 
import numpy as np 
import cv2

import util

IMG_DIR 		= './cropped_celeba/'
OUT_IMG_DIR_INT	= './results/gan_intermediate/'
OUT_IMG_DIR_TST	= './results/gan_test/'

IMG_NUM		= 20000
IMG_SIZE	= 64
BATCH_SIZE	= 50
EPOCH_NUM	= 40
G_TRAIN_NUM	= 2

class GAN(object):
	def __init__(self):
		self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
		self.labels = tf.placeholder(dtype=tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))

		self.g_outs 		= self.generator()
		self.d_outs_real 	= self.discriminator(self.labels)
		self.d_outs_fake 	= self.discriminator(self.g_outs, reuse=True)
		self.g_losses		= self.g_loss()
		self.d_losses		= self.d_loss()
		self.g_optim		= self.g_optimizer()
		self.d_optim		= self.d_optimizer()

	def generator(self):
		with tf.variable_scope('generator'):

			# down sampling
			y = tf.layers.conv2d(self.inputs, 
				filters=32, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [32,32,32]
			y = tf.layers.conv2d(y, 
				filters=64, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [16,16,64]
			y = tf.layers.conv2d(y, 
				filters=128, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [8,8,128]

			# up sampling
			y = tf.layers.conv2d_transpose(y, 
				filters=64, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.relu) # [16,16,64]
			y = tf.layers.conv2d_transpose(y, 
				filters=64, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.relu) # [32,32,64]
			y = tf.layers.conv2d_transpose(y, 
				filters=3, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.relu) # [64,64,3]

			return y

	def discriminator(self, x, reuse=None):
		with tf.variable_scope('discriminator', reuse=reuse):
			
			# down sampling
			y = tf.layers.conv2d(x, 
				filters=32, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [32,32,32]
			y = tf.layers.conv2d(y, 
				filters=64, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [16,16,64]
			y = tf.layers.conv2d(y, 
				filters=64, kernel_size=5, strides=(2,2), padding='same', activation=tf.nn.leaky_relu) # [8,8,64]

			# linear
			y = tf.reshape(y, [-1, 8*8*64])
			y = tf.layers.dense(y, 100, activation=tf.nn.relu)
			y = tf.layers.dense(y, 1, activation=tf.nn.sigmoid)

			return y

	def g_loss(self):
		alpha = 0.8
		discrim = tf.reduce_mean(-self._log(self.d_outs_fake))
		preserve = tf.losses.mean_squared_error(
				tf.reshape(self.g_outs, [-1, IMG_SIZE*IMG_SIZE*3]), 
				tf.reshape(self.labels, [-1, IMG_SIZE*IMG_SIZE*3])
			)
		return discrim + alpha*preserve

	def d_loss(self):
		return tf.reduce_mean(-self._log(self.d_outs_real)-self._log(1-self.d_outs_fake))/2

	def g_optimizer(self):
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		return tf.train.AdamOptimizer(1e-4).minimize(self.g_losses, var_list=g_vars)

	def d_optimizer(self):
		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		return tf.train.AdamOptimizer(1e-4).minimize(self.d_losses, var_list=d_vars)

	def _log(self, x):
		return tf.log(tf.maximum(1e-5, x))

def train(train_imgs, model, sess):
	n_train = len(train_imgs)
	for epoch in range(EPOCH_NUM):
		imgs = util.shuffle(train_imgs)
		g_loss = d_loss = 0.
		step = 0
		for start in range(0, n_train, BATCH_SIZE):
			end = min(start + BATCH_SIZE, n_train)
			batch = imgs[start:end]
			in_batch = util.get_input_imgs(batch)

			# train the discriminator
			dl, _ = sess.run([model.d_losses, model.d_optim],
				feed_dict={
					model.inputs: in_batch,
					model.labels: batch
				})
			d_loss += dl*(end-start)

			# train the generator
			for _ in range(G_TRAIN_NUM):
				o, gl, _ = sess.run([model.g_outs, model.g_losses, model.g_optim],
					feed_dict={
						model.inputs: in_batch,
						model.labels: batch
					})
				g_loss += gl*(end-start)/G_TRAIN_NUM

			if epoch%1==0 and step==0: # save outputs of 1st batch
				out_imgs = util.arrays2imgs(o[:5])
				for i, img in enumerate(out_imgs):
					cv2.imwrite(OUT_IMG_DIR_INT + 'e%d_%d.jpg' % (epoch, i), img)

			if step%100==0: 
				print('.. Step %5d, g_loss: %.5f, d_loss: %.5f' % (step, gl, dl))
			step += 1

		print('Epoch %3d >> avg_g_loss: %.5f \t avg_d_loss: %.5f' % (epoch, g_loss/n_train, d_loss/n_train))

	return model

def test(test_imgs, model, sess):
	in_imgs = util.get_input_imgs(test_imgs)
	outs, avg_g_loss, avg_d_loss = sess.run([model.g_outs, model.g_losses, model.d_losses],
		feed_dict={
			model.inputs: in_imgs,
			model.labels: test_imgs
		})
	print('Test >> avg_g_loss: %.5f \t avg_d_loss: %.5f' % (avg_g_loss, avg_d_loss))

	num_save = 50
	out_imgs = util.arrays2imgs(outs[:num_save])
	ori_imgs = util.arrays2imgs(test_imgs[:num_save])
	in_imgs = util.arrays2imgs(in_imgs[:num_save])
	for i in range(num_save):
		img = np.concatenate((out_imgs[i], ori_imgs[i], in_imgs[i]), axis=1)
		cv2.imwrite(OUT_IMG_DIR_TST + '0'*(2-len(str(i))) + '%d.jpg' % i, img)

def main():
	imgs = util.read_and_norm_imgs(IMG_DIR)
	train_imgs, test_imgs = util.train_test_split(imgs)

	print('Start training..')
	model = GAN()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	trained_model = train(train_imgs, model, sess)

	print('Start testing..')
	test(test_imgs, trained_model, sess)


if __name__ == '__main__':
	main()