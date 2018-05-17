import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
import os
import argparse
from sklearn import metrics

caffe.set_mode_gpu()

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mean', required=True, help='Path to mean image')
ap.add_argument('-d', '--deploy', required=True, help='Path to model prototxt')
ap.add_argument('-c', '--model', required=True, help='Path to caffemodel')
ap.add_argument('-t', '--test', required=True, help='Path to test dataset folder')
args = vars(ap.parse_args())

deploy_prototxt_file_path = args['deploy']
caffe_model_file_path = args['model']
test_lmdb_path = args['test']
mean_file_binaryproto = args['mean']

# Extract mean from the mean image file
mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
f = open(mean_file_binaryproto, 'rb')
mean_blobproto_new.ParseFromString(f.read())
mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
f.close()

# CNN reconstruction and loading the trained weights
net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

lmdb_env = lmdb.open(test_lmdb_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

test = []

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    
    label = int(datum.label)
    
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    
    out = net.forward_all(data=np.asarray([image]) - mean_image)
    plabel = int(out['prob'][0].argmax(axis=0))
   
    count += 1
    iscorrect = label == plabel
    correct += (1 if iscorrect else 0)

    test.append((label, plabel))

    matrix[(label, plabel)] += 1
    labels_set.update([label, plabel])

    print("Accuracy: {:.2f}%".format(100.*correct/count), end='\r')
    #if not iscorrect:
        # print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
        # sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        # sys.stdout.flush()

print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")

csv_path = os.path.split(args['model'])[0]
with open(csv_path + '/prediction_eval.csv', 'w') as f:
    f.write('true, pred\n')
    for t in test:
        f.write('{},{}\n'.format(str(t[0]), str(t[1])))
print('Saved prediction_eval.csv to {}'.format(csv_path))
f.close()

t_true = [t[0] for t in test]
t_pred = [t[1] for t in test]

confusion_matrix = metrics.confusion_matrix(t_true, t_pred)
report = metrics.classification_report(t_true, t_pred)
accuracy = metrics.accuracy_score(t_true, t_pred)
print('Confusion Matrix\n')
print(confusion_matrix, end='\n\n')
print(report, end='\n\n')
print('Accuracy: {:.4f}'.format(accuracy))
# print("")
# print("Confusion matrix:")
# print("(r , p) | count")
# for l in labels_set:
#     for pl in labels_set:
#         print("(%i , %i) | %i" % (l, pl, matrix[(l,pl)]))