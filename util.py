import cv2
import glob
import numpy as np

def read_and_norm_imgs(img_dir, img_num=20000):
	return np.array([cv2.imread(file) for file in glob.glob(img_dir + '*.jpg')[:img_num]]) / 255.

def train_test_split(imgs, train_ratio=0.9):
	n_train = int(train_ratio * len(imgs))
	return imgs[:n_train], imgs[n_train:]

def shuffle(imgs):
	return np.array([imgs[i] for i in np.random.permutation(len(imgs))])

def norm_imgs(imgs):
	return imgs / 255.

def arrays2imgs(arrays):
	return (arrays * 255.).astype(np.uint8)

def get_input_imgs(imgs):
	return np.array([_draw_box(img) for img in imgs])

def _draw_box(img):
	return cv2.rectangle(img.copy(), (12,12), (52,52), (0,0,0), cv2.FILLED)