import csv
import glob
from sklearn.metrics import confusion_matrix, roc_auc_score
from caffe_toolkit import *

image_dir = '/path/to/spectrograms'
model_dir = '/path/to/the/caffe/model'
batch_size = 100
use_gpu = True

caffemodel = None
deploy_file = None
mean_file = None
labels_file = None

for filename in os.listdir(model_dir):
    full_path = os.path.join(model_dir, filename)
    if filename.endswith('.caffemodel'):
        caffemodel = full_path
    elif filename == 'deploy.prototxt':
        deploy_file = full_path
    elif filename.endswith('.binaryproto'):
        mean_file = full_path
    elif filename == 'labels.txt':
        labels_file = full_path

net = get_net(caffemodel, deploy_file, use_gpu)
transformer = get_transformer(deploy_file, mean_file)
_, channels, height, width = transformer.inputs['data']

if channels == 3:
    mode = 'RGB'
elif channels == 1:
    mode = 'L'
else:
    raise ValueError('Invalid number for channels: %s' % channels)

image_files_fullpath = glob.glob(image_dir + '**/*.png', recursive=True)
images = [load_image(image_file, height, width, mode) for image_file in image_files_fullpath]
labels = read_labels(labels_file)
scores = forward_pass(images, net, transformer, batch_size=batch_size)

image_files = [os.path.splitext(os.path.basename(path))[0] for path in image_files_fullpath]
results = dict(zip(image_files, scores[:,1]))

def read_csv_to_dict(path):
    d = {}
    with open(path, 'r') as f:
        next(f)  # header
        reader = csv.reader(f)
        for name, value in reader:
            d[name] = value
    return d

groundtruth_filepath = '/home/tracek/Data/Birdman/metadata.csv'
groundtruth = read_csv_to_dict(groundtruth_filepath)

y_true = np.zeros(len(results))
y_pred = np.zeros(len(results))

for idx, (name, result) in enumerate(results.items()):
    name_stripped = name.replace('ff1010bird_', '').replace('warblrb10k_public_', '')
    result_binary = float(result) >= 0.5
    truth = int(groundtruth[name_stripped])
    y_pred[idx] = result_binary
    y_true[idx] = truth == 1

cm = confusion_matrix(y_true, y_pred)
tn = cm[0][0]
fn = cm[0][1]
tp = cm[1][1]
fp = cm[1][0]

accuracy = (tp + tn) / (tp + tn + fn + fp)
precision = tp/(tp + fp)
recall = tp/(tp + fn)

rocauc = roc_auc_score(y_true, y_pred)

print('Accuracy: {:2f}'.format(accuracy))
print('Precision: {:2f}'.format(precision))
print('Recall: {:2f}'.format(recall))
print('AUC: {:2f}'.format(rocauc))

print("Done!")