import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def reshape_raw_images(raw_images):
    """
    gets a set of flattened images in dimensions (set_size, 3072 (=3*32*32))
    where every row is a flattened image organized by RGB,
     i.e first 1024 cells are red, 1024 next are green...

    it returns a set of flattened images in dimensions (set_size, 3072 (=32*32*3))
    where every row is a flattened image organized by pixels,
     i.e first 3 cells represent the RGB values of the first pixel
    """
    images_num = raw_images.shape[0]
    images_in_3d = np.reshape(raw_images, (images_num, 3, 32, 32))
    # reshape to 32*32*3
    images_in_3d = np.transpose(images_in_3d, (0, 2, 3, 1))
    images = np.reshape(images_in_3d, (images_num, 3072))
    return np.copy(images)


def get_data_from_batches(data_folder, batch_type):
    raw_images = []
    labels = []
    for file in os.listdir(data_folder):
        if '_'.join((batch_type, 'batch')) in file:
            data_dict = unpickle(os.path.join(data_folder, file))
            raw_images.append(data_dict['data'].astype('float'))
            labels.append(np.array(data_dict['labels']))

    raw_images_set = np.vstack(raw_images)
    images_set = reshape_raw_images(raw_images_set)
    images_labels = np.concatenate(labels)
    return images_set, images_labels


def get_data(data_folder):
    training_images, training_labels = get_data_from_batches(data_folder, 'data')
    testing_images, testing_labels = get_data_from_batches(data_folder, 'test')
    return training_images, training_labels, testing_images, testing_labels


def subsample_data(training_images, training_labels, testing_images, testing_labels, training_num=5000,
                   testing_num=500):
    """
    the data is sampled using masks since by integer indexing we get a deep copy of the data sets
    """
    training_mask = range(training_num)
    testing_mask = range(testing_num)
    return (training_images[training_mask], training_labels[training_mask],
            testing_images[testing_mask], testing_labels[testing_mask])


def normalize_data(data):
    mean = np.mean(data)
    return data - mean


def show_image(image):
    image = image.reshape((32, 32, 3))
    plt.imshow(image.astype('uint8'))
    plt.show()
