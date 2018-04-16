import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import glob
import argparse
import random
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True, help='Path to dataset directory')
args = vars(ap.parse_args())

# Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, equalize=False):
    """Function that resize an image and equalize it if necessary."""
    if equalize:
        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

path = args['path']
parent_path = os.path.sep.join(path.split(os.path.sep)[:-1])
sibling_path = path.split(os.path.sep)[-1] + '_lmdb'
sibling_path = os.path.sep.join([parent_path, sibling_path])
train_lmdb = os.path.sep.join([sibling_path, 'train'])
validation_lmdb = os.path.sep.join([sibling_path, 'validation'])

if not os.path.exists(sibling_path):
    os.makedirs(sibling_path)

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

dataset = []
for r, dirs, files in os.walk(path):
    if len(dirs) > 0:
        labels = dirs
        continue  # use only leaf folders
    files_full_path = ['{}/{}'.format(r, f) for f in files]
    directory_name = r.split(os.path.sep)[-1]
    dataset.append((files_full_path, directory_name))

label_dict = {
    'basalcellcarcinoma': 0,
    'lentigo': 1,
    'malignantmelanoma': 2,
    'pigmentednevus': 3,
    'seborrheickeratosis': 4,
    'wart': 5
}

X = [(img, label) for ndataset, label in dataset for img in ndataset]
y = [label_dict[label] for _, label in X]
# Shuffle dataset
random.shuffle(X)

train_data, test_data, _, _ = train_test_split(X, y, train_size=0.8, stratify=y)

print('Creating train_lmdb...')

train_time = time()
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, (img_path, label) in enumerate(train_data):
        if in_idx % 100 == 0:
            print('Processed {}/{}'.format(in_idx, len(train_data)), end='\r')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        num_label = label_dict[label]
        datum = make_datum(img, num_label)

        key = '{:0>6d}'.format(in_idx)
        in_txn.put(key.encode(), datum.SerializeToString())
        # print '{:0>6d}'.format(in_idx) + ':' + img_path
in_db.close()
print('Finished {} train_lmdb in {:.2f} sec'.format(len(train_data), (time() - train_time)))


print('\nCreating validation_lmdb...')

test_time = time()
in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    old_t = time()
    for in_idx, (img_path, label) in enumerate(test_data):
        if in_idx % 100 == 0:
            print('Processed {}/{}'.format(in_idx, len(test_data)), end='\r\r')

            old_t = time()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        num_label = label_dict[label]
        datum = make_datum(img, num_label)

        key = '{:0>6d}'.format(in_idx)
        in_txn.put(key.encode(), datum.SerializeToString())
        # in_txn.put('{:0>6d}'.format(in_idx), datum.SerializeToString())
        # print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()
print('Finished {} test_lmdb in {:.2f} sec'.format(len(test_data), (time() - test_time)))

print('\nFinished processing all images in {:.2f}'.format(time() - train_time))
