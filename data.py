from logging import raiseExceptions
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
import os
import re
import os.path as osp
from utils.tf_record import *
from floorplan_room_type import *
from utils.rgb_ind_convertor import *

CURR_DIR = osp.dirname(__file__) #TF2DeepFloorplan/

TRAIN_FILE_LIST_FN = 'r3d_train.txt'
TFRECORD_FN = 'r3d.tfrecords'

def convert_one_hot_to_image(one_hot,dtype='float',act=None):
    if act=='softmax':
        one_hot = tf.keras.activations.softmax(one_hot)
    [n,h,w,c] = one_hot.shape.as_list()
    im=tf.reshape(tf.keras.backend.argmax(one_hot,axis=-1),
                  [n,h,w,1])
    if dtype=='int':
        im = tf.cast(im,dtype=tf.uint8)
    else:
        im = tf.cast(im,dtype=tf.float32)
    return im

def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'boundary':tf.io.FixedLenFeature([],tf.string),
              'room':tf.io.FixedLenFeature([],tf.string),
              'door':tf.io.FixedLenFeature([],tf.string)}
    return tf.io.parse_single_example(example_proto,feature)

def decodeAllRaw(x):
    image = tf.io.decode_raw(x['image'],tf.uint8)
    boundary = tf.io.decode_raw(x['boundary'],tf.uint8)
    room = tf.io.decode_raw(x['room'],tf.uint8)
    return image,boundary,room

def preprocess(img,bound,room,num_classes,size=512):
    img = tf.cast(img,dtype=tf.float32)
    img = tf.reshape(img,[-1,size,size,3])/255
    bound = tf.reshape(bound,[-1,size,size])
    room = tf.reshape(room,[-1,size,size])
    hot_b = tf.one_hot(bound,3,axis=-1)
    hot_r = tf.one_hot(room,num_classes,axis=-1) # len(floorplan_room_map) = 9 or len(ROOM_TYPE) = 31
    return img,bound,room,hot_b,hot_r


def loadDataset(size=512):
    raw_dataset = tf.data.TFRecordDataset('r3d.tfrecords')
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def make_path_list_txt(path,txt_name):
	file_list = os.listdir(path)

	id_list = set()
	for fn in file_list:
		id = re.split(r'[_.]',fn)
		if len(id)>1:
			if id[1] not in ['close','original','room','rooms','multi','wall','jpg','png']:
				id[0] += '_' + id[1]
		id_list.add(id[0])
	
	f = open(txt_name, 'w')
	for id in id_list:
		wall = '_wall.png'
		close = '_close.png'
		room = '_rooms.png'
		close_wall = '_close_wall.png'

		img = path + id + '.png'
		if os.path.exists(img):
			f.write(img)
		else:
			f.write(path + '/' + id + '.jpg')
		f.write('\t' + path + id + wall)
		f.write('\t' + path + id + close)
		f.write('\t' + path + id + room)
		f.write('\t' + path + id + close_wall + '\n')
	f.close()

def updateDataset():
    print('\nnew dataset updating...\n')
    try :
        dataset_dir = osp.join(CURR_DIR, 'dataset/')
        if not osp.isdir(dataset_dir):
            raise Exception('NO DATASET FILES. PLEASE DOWNLOAD FROM LINK.')

        train_file_path = osp.join(CURR_DIR,TRAIN_FILE_LIST_FN)
        make_path_list_txt(os.path.join(dataset_dir,'train/'), train_file_path)
        train_paths = open(train_file_path, 'r').read().splitlines()
        
        tfrecord_path = osp.join(CURR_DIR, TFRECORD_FN)
        write_bd_rm_record(train_paths, tfrecord_path)
    except Exception as e:
        print('[error]',e)
        exit()


if __name__ == "__main__":
    dataset = loadDataset()
    for ite in range(2):
        for data in list(dataset.shuffle(400).batch(1)):
            img,bound,room = decodeAllRaw(data)
            img,bound,room,hb,hr = preprocess(img,bound,room)
            plt.subplot(1,3,1);plt.imshow(img[0].numpy())
            plt.subplot(1,3,2);plt.imshow(bound[0].numpy())
            plt.subplot(1,3,3);plt.imshow(convert_one_hot_to_image(hb)[0].numpy());plt.show()


            pdb.set_trace()    
            break


    
