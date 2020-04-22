from classifiers.knn import KNN
from data_utils import get_data, subsample_data
from classification_utils import cross_validate_params, get_success_rate


CIFAR10_FOLDER = 'cifar10_data'
K_CHOICES = (1, 3, 5, 8, 10, 12, 15, 20, 50, 100)


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels)
    knn = KNN()
    best_params = cross_validate_params(knn, training_set, training_labels, k=K_CHOICES)
    knn.train(training_set, training_labels)
    result_labels = knn.predict(testing_set, **best_params)
    success_rate = get_success_rate(result_labels, testing_labels)
    print(f'for best params {best_params}, success rate is: {success_rate}')


if __name__ == '__main__':
    main()
