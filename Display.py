from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import os
import sys
import math
import copy

import numpy as np
import pandas as pd
import tensorflow as tf

from Model import Model
from Preprocessing import *

BATCH_SIZE = 12

def generate_dataset(paths):
	X_train = []
	y_train = []
	for path in paths:
		X_train, y_train = generate_one_dataset(X_train, y_train, path)
	X_train, X_val, y_train, y_val = split_dataset(X_train, y_train, 0.2)
	mean, std = get_stats_of_images(X_train)
	return X_train, X_val, y_train, y_val, mean, std

def generate_one_dataset(X_train, y_train, path):
	files = fetch_files(path)
	angles = fetch_angles(path, files)
	X_train.extend(files)
	y_train.extend(angles)
	return X_train, y_train

def fetch_files(path):
	image_path = os.path.join(path, "center")
	files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
			if os.path.isfile(os.path.join(image_path, f))
		]
	files.sort()
	return files

def fetch_angles(path, files):
	data = pd.read_csv(os.path.join(path, "interpolated.csv"))
	data = data[['timestamp','frame_id', 'filename', 'angle']]
	data = data.loc[data['frame_id'] == 'center_camera']
	angles = []
	for f in files:
		frame = os.path.basename(f)
		angle = (data.loc[data['timestamp'] == 
				 int(frame[:-4]), ['angle']].values[0][0])
		angles.append(angle)
	return angles

def display(files, labels, model):
	for i in range(len(files)):
		image = load_image(files[i])
		eval_image = load_image(files[i])
		eval_image = mean_subtraction(eval_image, model.get_mean())
		eval_image = normalize(eval_image, model.get_std())
		if labels != None:
			image = draw_angle(image, labels[i], "TRUTH")
		if model != None:
			input_image = np.array([eval_image])
			pred_angle = model.predict(input_image)
			print(pred_angle)	
			pred_angle = pred_angle[0][0]	
			image = draw_angle(image, pred_angle, "PREDICT")
		cv2.imshow("Video", image)
		cv2.waitKey(17)
	cv2.destroyAllWindows()

def draw_angle(image, angle, meaning="TRUTH"):
	if meaning == "TRUTH":
		color = (0, 255, 0)
	else:
		color = (255, 0, 0)
	offset = 5
	height = image.shape[0]
	width = image.shape[1]
	point_a = (int(width/2), height - offset)
	c = 200
	a = int(c * math.sin(angle))
	b = int(c * math.cos(angle))
	point_b = (int(width/2) - a, (height - offset - b))
	cv2.line(image, point_a, point_b, color=color, thickness=2)
	return image

if __name__ == "__main__":
	paths = ["data/output/1"]#, 
	 	#"data/output/2", 
	 	#"data/output/4",
	 	#"data/output/5",
	 	#"data/output/6"]
	X_train, X_val, y_train, y_val, mean, std = generate_dataset(paths)
	sess = tf.Session()
	network = Model(sess, X_train, y_train, X_val, y_val, mean=mean, std=std)
	if sys.argv[1] == "train":
		network.train()
		user_input = raw_input("PRESS ENTER TO CONTINUE")
	else:
		network.load("model.pkl")
	display(X_train, y_train, network)
	#test_path = "Ch2_001"
	#test = fetch_files(test_path)
	#display(test, None, network)