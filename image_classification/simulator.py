from classifiers.knn import KNN
from data_utils import get_data, subsample_data
from classification_utils import cross_validate, get_success_rate


CIFAR10_FOLDER = 'cifar10_data'
k_choices = (1, 3, 5, 8, 10, 12, 15, 20, 50, 100)


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels,
                                                   500, 50)
    knn = KNN()
    best_params = cross_validate(knn, training_set, training_labels, k=k_choices)
    knn.train(training_set, training_labels)
    result_labels = knn.predict(testing_set, **best_params)
    success_rate = get_success_rate(result_labels, testing_labels)
    print(f'for best params {best_params}, success rate is: {success_rate}')


if __name__ == '__main__':
    main()
