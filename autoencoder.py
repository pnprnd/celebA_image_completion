import tensorflow as tf 
import numpy as np 
import cv2

import util

IMG_DIR 		= './cropped_celeba/'
OUT_IMG_DIR_INT	= './aen_generated_images_intermediate/'
OUT_IMG_DIR_TST	= './aen_generated_images_test/'

IMG_NUM		= 20000
IMG_SIZE	= 64
BATCH_SIZE	= 50
EPOCH_NUM	= 30

class Autoencoder(object):
	def __init__(self):
		self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
		self.labels = tf.placeholder(dtype=tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))

		self.outs 	= self.forward()
		self.losses = self.loss()
		self.optim	= self.optimizer()

	def forward(self):

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

	def loss(self):
		return tf.losses.mean_squared_error(
				tf.reshape(self.outs, [-1, IMG_SIZE*IMG_SIZE*3]), 
				tf.reshape(self.labels, [-1, IMG_SIZE*IMG_SIZE*3])
			)

	def optimizer(self):
		return tf.train.AdamOptimizer(1e-4).minimize(self.losses)

def train(train_imgs, model, sess):
	n_train = len(train_imgs)
	for epoch in range(EPOCH_NUM):
		imgs = util.shuffle(train_imgs)
		loss = 0.
		step = 0
		for start in range(0, n_train, BATCH_SIZE):
			end = min(start + BATCH_SIZE, n_train)
			batch = imgs[start:end]
			o, l, _ = sess.run([model.outs, model.losses, model.optim],
				feed_dict={
					model.inputs: util.get_input_imgs(batch),
					model.labels: batch
				})
			loss += l*(end-start)

			if epoch%2==0 and step==0: # save outputs of 1st batch
				out_imgs = util.arrays2imgs(o[:5])
				for i, img in enumerate(out_imgs):
					cv2.imwrite(OUT_IMG_DIR_INT + 'e%d_%d.jpg' % (epoch, i), img)

			if step%100==0: 
				print('.. Step %5d, loss: %.5f' % (step, l))
			step += 1

		print('Epoch %3d >> avg_loss: %.5f' % (epoch, loss/n_train))

	return model

def test(test_imgs, model, sess):
	in_imgs = util.get_input_imgs(test_imgs)
	outs, avg_loss = sess.run([model.outs, model.losses],
		feed_dict={
			model.inputs: in_imgs,
			model.labels: test_imgs
		})
	print('Avg testing loss: %.5lf' % avg_loss)

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
	model = Autoencoder()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	trained_model = train(train_imgs, model, sess)

	print('Start testing..')
	test(test_imgs, trained_model, sess)


if __name__ == '__main__':
	main()