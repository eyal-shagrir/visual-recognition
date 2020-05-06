from classifiers.knn import KNN
from classifiers.svm import SVM
from data_utils import get_data, subsample_data, normalize_data, show_image
from classification_utils import get_success_rate, predict_success_rate, cross_validate_and_predict

CIFAR10_FOLDER = 'cifar10_data'


K_CHOICES = (1, 3, 5, 8, 10, 12, 15, 20, 50, 100)

CLASSIFIERS_PARAMETERS = {KNN.__name__: {'k': K_CHOICES}}


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels,
                                                   training_num=1000, testing_num=100)
    normalize_data(training_set, testing_set)
    svm = SVM()
    W = svm.train(training_set, training_labels)
    result_labels = svm.predict(testing_set, W)
    success_rate = get_success_rate(result_labels, testing_labels)
    print(f'for {type(svm).__name__} success rate is: {success_rate}')


    # knn = KNN()
    # cross_validate_and_predict(knn, training_set, training_labels, testing_set, testing_labels, k=K_CHOICES)


if __name__ == '__main__':
    main()
