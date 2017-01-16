import os
import scipy.io.wavfile as wav
import numpy as np
import glob
import multiprocessing
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import firwin, lfilter
import time
import csv
from skimage import transform

root_path = '/waves/*.wav'
output_dir = '/somedir'
metadata_path = '/labels/dir'

all_recordings = glob.glob(root_path)

with open(metadata_path) as mf:
    reader = csv.reader(mf)
    header = next(reader)
    metadata = dict((rows[0],rows[1]) for rows in reader)


def plot_spectrogram(data, rate, name, dir_label):
    window_size = 2 ** 10
    overlap = window_size // 8
    window = sig.tukey(M=window_size, alpha=0.25)

    freq, time, spectrogram = sig.spectrogram(data, fs=rate, window=window, nperseg=window_size, scaling='density', noverlap=overlap)
    spectrogram = np.log10(np.flipud(spectrogram))
    try:
        if spectrogram.shape[1] > 512:
            spec_padded = spectrogram[:512,:512]
        elif spectrogram.shape[1] < 512:
            spec_padded = np.pad(spectrogram, ((0, 0), (0, 512 - spectrogram.shape[1])), mode='median')[:512, :]
        else:
            spec_padded = spectrogram
    except Exception as e:
        print('ERROR!')
        print('Fault in: {}'.format(name))
        raise
    spec_padded = transform.downscale_local_mean(spec_padded, (2, 2))

    final_path = os.path.join(output_dir, dir_label, name + '.png')
    plt.imsave(final_path, spec_padded, cmap=plt.get_cmap('gray'))


def run(path):
    print('Processing {}'.format(path))
    directory, filename = os.path.split(path)
    subdirectory = directory[directory.rfind('/') + 1:]
    filename_noext = os.path.splitext(filename)[0]
    name = subdirectory + '_' + filename_noext
    dir_label = metadata[filename_noext]

    (rate, sample) = wav.read(path)
    plot_spectrogram(sample, rate, name, dir_label)



start = time.time()

unique_labels = set(metadata.values())

for label in unique_labels:
    p = os.path.join(output_dir, label)
    if not os.path.exists(p):
        os.makedirs(p)

pool = multiprocessing.Pool(4)
pool.map(run, all_recordings)
print('Total time: {}'.format(time.time() - start))

print('Done')