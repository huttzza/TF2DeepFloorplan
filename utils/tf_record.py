import numpy as np

import tensorflow as tf

#from scipy.misc import imread, imresize, imsave
from imageio import imread
from matplotlib import pyplot as plt
from utils.rgb_ind_convertor import *
import cv2

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_bd_rm_images(path):
	paths = path.split('\t')
	
	#paths = [os.getcwd() + '/dataset/'+ p for p in paths]
	image = imread(paths[0], pilmode='RGB')
	close = imread(paths[2], pilmode='L')
	room  = imread(paths[3], pilmode='RGB')
	close_wall = imread(paths[4], pilmode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
	image = cv2.resize(image, (512, 512))
	close = cv2.resize(close, (512, 512)) / 255.
	close_wall = cv2.resize(close_wall, (512, 512)) / 255.
	room = cv2.resize(room, (512, 512))

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)

	cw_ind[cw_ind==1] = 2
	cw_ind[d_ind==1] = 1

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)
	cw_ind = cw_ind.astype(np.uint8)

	return image, cw_ind, room_ind, d_ind

def write_bd_rm_record(paths, name='r3d.tfrecords'):
	writer = tf.io.TFRecordWriter(name)
	
	for i in range(len(paths)):
		# Load the image
		image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
					'boundary': _bytes_feature(tf.compat.as_bytes(cw_ind.tostring())),
					'room': _bytes_feature(tf.compat.as_bytes(room_ind.tostring())),
					'door': _bytes_feature(tf.compat.as_bytes(d_ind.tostring()))}
		
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))
    
		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
    
	writer.close()

