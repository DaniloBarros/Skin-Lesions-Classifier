import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import argparse
import matplotlib.pyplot as plt

caffe.set_mode_gpu() 

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mean', required=True, help='Path to mean image')
ap.add_argument('-d', '--deploy', required=True, help='Path to model prototxt')
ap.add_argument('-c', '--model', required=True, help='Path to caffemodel')
ap.add_argument('-t', '--test', required=True, help='Path to test dataset folder')
args = vars(ap.parse_args())

#Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT, equalize=False):

    if equalize:
        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(args['mean'], 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
# net = caffe.Net('/home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_2/caffenet_deploy_2.prototxt',
#                 '/home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_2/caffe_model_2_iter_10000.caffemodel',
#                 caffe.TEST)

net = caffe.Net(args['deploy'],
                args['model'],
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))


'''
Making predicitions
'''
##Reading image paths
path = args['test']

dataset = []
for r, dirs, files in os.walk(path):
    if len(dirs) > 0:
        labels = dirs
        continue  # use only leaf folders
    files_full_path = ['{}/{}'.format(r, f) for f in files]
    directory_name = r.split(os.path.sep)[-1]
    dataset.append((files_full_path, directory_name))

test_dataset = [(img, label) for ndataset, label in dataset for img in ndataset]
label_dict = {
    'ak': 10,
    'basalcellcarcinoma': 5,
    'dermatofibroma': 11,
    'hemangioma': 3,
    'intraepithelial_carcinoma': 0,
    'lentigo': 7,
    'melanoma': 4,
    'naevus': 8,
    'pyogenic_granuloma': 2,
    'scc': 9,
    'seborrheickeratosis': 1,
    'wart': 6
}

test_ids = []

#Making predictions
for in_idx, (img_path, label) in enumerate(test_dataset):
    print('Processed {}/{}'.format(in_idx, len(test_dataset)), end='\r')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    # print(label_dict[label], out['loss'], out['acc/top-1'], out['acc/top-5'])

    img_id = os.path.sep.join(img_path.split(os.path.sep)[-2:])
    str_label = label
    true_label = label_dict[str_label]
    softmaxwithloss = out['prob'][0]
    print(softmaxwithloss, softmaxwithloss.argmax())
    pred = softmaxwithloss.argmax()
    test = (img_id, str_label, true_label, pred)
    test_ids.append(test)

    print(test)
    print('-------')


'''
Making submission file
'''
with open(args['test']+"/prediction.csv","w") as f:
    f.write("id,label,true,prob\n")
    for t in test_ids:
        f.write(','.join([str(i) for i in t])+"\n")
f.close()
