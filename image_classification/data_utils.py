import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(data_folder):
    training_batches = []
    training_labels_batches = []
    for file in os.listdir(data_folder):
        if 'data_batch' in file:
            data_dict = unpickle(os.path.join(data_folder, file))
            training_batches.append(data_dict[b'data'].astype('float'))
            training_labels_batches.append(np.array(data_dict[b'labels']))
        elif 'test_batch' in file:
            data_dict = unpickle(os.path.join(data_folder, file))
            testing_images = (data_dict[b'data'].astype('float'))
            testing_labels = np.array(data_dict[b'labels'])

    training_images = np.vstack(training_batches)
    training_labels = np.concatenate(training_labels_batches)
    return training_images, training_labels, testing_images, testing_labels
