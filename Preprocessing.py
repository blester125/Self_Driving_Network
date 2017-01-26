from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
from sklearn.cross_validation import train_test_split

def load_image(path):
	image = cv2.imread(path)
	return image

def load_images(paths):
	images = []
	for path in paths:
		image = load_image(path)
		images.append(image)
	images = np.array(images)
	return images

def mean_subtraction(data, mean):
	data = data.astype(mean.dtype)
	data -= mean
	return data

def normalize(data, stddev):
	data = data.astype(stddev.dtype)
	data /= stddev
	return data

def split_dataset(X_train, y_train, amount=0.2):
	return train_test_split(X_train, y_train, test_size=amount, random_state=0)

def load_minibatch(
		X_train, 
		y_train, 
		offset, 
		batch_size, 
		mean=0, 
		std=1, 
		images=False):
	y_batch = np.array(y_train[offset:min(offset + batch_size, len(X_train))])
	if images:
		X_batch = []
		for i in range(batch_size):
			if offset + i >= len(X_train):
				break
			image = load_image(X_train[offset+i])
			X_batch.append(image)
		X_batch = np.array(X_batch)
	else:
		X_batch = np.array(
					X_train[offset:min(offset + batch_size, len(X_train))])
	X_batch = mean_subtraction(X_batch, mean)
	X_batch = normalize(X_batch, std)
	return X_batch, y_batch

def get_stats(data, piece_wise=False):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	return mean, std

def get_stats_of_images(paths):
	mean = 0
	std = 0
	for path in paths:
		image = load_image(path)
		mean += np.mean(image)
		std += np.std(image)
	mean = mean / len(paths)
	std = std / len(paths)
	return np.array(mean), np.array(std)

def read_CSV():
	pass

def write_CSV():
	pass

def generate_dataset():
	pass


if __name__ == "__main__":
	names = ['1.JPG', '1.JPG', '1.JPG', '1.JPG']
	test = load_image(names[0])
	print(type(test))
	print(test.shape)
	tests = load_images(names)
	print(type(tests))
	print(tests.shape)