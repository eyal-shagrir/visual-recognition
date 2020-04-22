from classifiers.knn import KNN
from data_utils import get_data

import numpy as np
from matplotlib import pyplot as plt

CIFAR10_FOLDER = 'cifar10_data'


def subsample_data(training_images, training_labels, testing_images, testing_labels, training_num=5000,
                   testing_num=500):
    """
    the data is sampled using masks since by integer indexing we get a deep copy of the data sets
    """
    training_mask = range(training_num)
    testing_mask = range(testing_num)
    return (training_images[training_mask], training_labels[training_mask],
            testing_images[testing_mask], testing_labels[testing_mask])


def get_success_rate(labels, y_test):
    num_tests = y_test.shape[0]
    num_successes = np.sum(labels == y_test)
    success_rate = num_successes / num_tests
    return success_rate


def show_success_rate_by_k(classifier, testing_set, testing_labels, k_limit=21):
    classifier_name = type(classifier).__name__
    success_rates = []
    for k in range(1, k_limit):
        result_labels = classifier.predict(testing_set, k=k)
        success_rate = get_success_rate(result_labels, testing_labels)
        success_rates.append(success_rate)

        # print(f'{classifier_name} with k = {k} success rate: {success_rate}')
        # show_sampled_images(testing_images, result_labels, samples_per_class=1)

    plt.plot(range(1, k_limit), success_rates, '-bo', label='success rate')
    plt.xlabel('k')
    plt.title(classifier_name)
    plt.legend()
    plt.show()


def divide_to_folds(training_set, training_labels, fold_size, validation_fold_idx):
    idx = validation_fold_idx * fold_size
    after_idx = (validation_fold_idx + 1) * fold_size
    training_folds = np.vstack((training_set[: idx], training_set[after_idx:]))
    training_folds_labels = np.concatenate((training_labels[: idx], training_labels[after_idx:]))
    validation_fold = training_set[idx: after_idx]
    validation_fold_labels = training_labels[idx: after_idx]
    return training_folds, training_folds_labels, validation_fold, validation_fold_labels


def knn_cross_validate(knn, training_set, training_labels,
                       num_folds=5, k_choices=(1, 3, 5, 8, 10, 12, 15, 20, 50, 100)):
    training_size = training_set.shape[0]
    fold_size = int(training_size / num_folds)

    success_results = np.zeros((len(k_choices), num_folds))

    for k_index, k in enumerate(k_choices):
        for i in range(num_folds):
            (training_folds, training_folds_labels,
             validation_fold, validation_fold_labels) = divide_to_folds(training_set, training_labels, fold_size, i)

            knn.train(training_folds, training_folds_labels)

            result_labels = knn.predict(validation_fold, k=k)
            success_rate = get_success_rate(result_labels, validation_fold_labels)
            success_results[k_index, i] = success_rate

    averaged_success_rates = np.average(success_results, axis=1)
    plt.plot(k_choices, averaged_success_rates, '-bo', label='success rate')
    plt.xlabel('k')
    plt.title('cross-validation for k')
    plt.legend()
    plt.show()
    max_k_idx = int(np.argmax(averaged_success_rates))
    return k_choices[max_k_idx]


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels)
    knn = KNN()
    knn.train(training_images, training_labels)
    k = knn_cross_validate(knn, training_set, training_labels)
    knn.train(training_set, training_labels)
    result_labels = knn.predict(testing_set, k=k)
    success_rate = get_success_rate(result_labels, testing_labels)
    print(f'for best k={k}, success rate is: {success_rate}')


if __name__ == '__main__':
    main()
