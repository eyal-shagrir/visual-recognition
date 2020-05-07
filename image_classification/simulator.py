from classifiers.knn import KNN
from classifiers.svm import SVM
from data_utils import get_data, subsample_data, normalize_data, show_image
from classification_utils import predict_success_rate, cross_validate_and_predict

CIFAR10_FOLDER = 'cifar10_data'

# KNN HYPERPARAMETERS
Ks = (1, 3, 5, 8, 10, 12, 15, 20, 50, 100)  # best is around 10
KNN_HP = {'k': Ks}

# SVM HYPERPARAMETERS
ALPHAS = (1e-11, 1e-7, 1e-3)  # best is around 30000
REGS = (1000, 5000, 10000, 20000, 30000)  # best is around 30000
ITERS = (100, 300, 600, 900)  # best is around 700
SVM_HP = {'alpha': ALPHAS, 'reg': REGS, 'iterations': ITERS}

CLASSIFIERS_HP = {'KNN': KNN_HP, 'SVM': SVM_HP}


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels,
                                                   training_num=5000, testing_num=500)
    normalize_data(training_set, testing_set)

    for classifier_name, classifier_cls in {'KNN': KNN, 'SVM': SVM}.items():
        classifier = classifier_cls()
        cross_validate_and_predict(classifier,
                                   training_set, training_labels,
                                   testing_set, testing_labels,
                                   **CLASSIFIERS_HP[classifier_name])


if __name__ == '__main__':
    main()
