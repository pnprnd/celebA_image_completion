import cv2
import glob
import numpy as np
import face_recognition as fr

ORI_IMG_DIR = './img_align_celeba/'
RES_IMG_DIR = './cropped_celeba/'
IMG_NUM		= 20000
IMG_SIZE	= 64

def main():
	# read raw images
	imgs = [np.array(cv2.imread(file)) 
		for file in glob.glob(ORI_IMG_DIR + '*.jpg')[:IMG_NUM*2]]
	print('Done reading files')

	# crop & resize
	c = 0
	for img in imgs:
		locations = fr.face_locations(img)
		if not locations: continue
		bounding_box = locations[0]
		cropped = img[bounding_box[0]:bounding_box[2], bounding_box[3]:bounding_box[1]]
		resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
		cv2.imwrite(RES_IMG_DIR + '0'*(5-len(str(c))) + str(c) + '.jpg', resized)
		c += 1
		if c%1000 == 0: print('Processed %d images' % c)
		if c >= IMG_NUM: break

if __name__ == '__main__':
	main()