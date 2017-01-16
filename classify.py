import glob
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

with open('/home/tracek/Data/Birdman/submission_tracewski_bandpass_binary.csv', 'w') as f:
    for key, value in results.items():
        f.write('{0},{1:.3f},{2}\n'.format(key, value, 1 if value>=0.5 else 0))

with open('/home/tracek/Data/Birdman/submission_tracewski_bandpass.csv', 'w') as f:
    for key, value in results.items():
        f.write('{0},{1:.3f}\n'.format(key, value))

print("Done!")