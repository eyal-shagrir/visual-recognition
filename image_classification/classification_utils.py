import numpy as np
from matplotlib import pyplot as plt


def get_success_rate(labels, y_test):
    num_tests = y_test.shape[0]
    num_successes = np.sum(labels == y_test)
    success_rate = num_successes / num_tests
    return success_rate


def divide_to_folds(training_set, training_labels, fold_size, validation_fold_idx):
    idx = validation_fold_idx * fold_size
    after_idx = (validation_fold_idx + 1) * fold_size
    training_folds = np.vstack((training_set[: idx], training_set[after_idx:]))
    training_folds_labels = np.concatenate((training_labels[: idx], training_labels[after_idx:]))
    validation_fold = training_set[idx: after_idx]
    validation_fold_labels = training_labels[idx: after_idx]
    return training_folds, training_folds_labels, validation_fold, validation_fold_labels


def show_parameter_success_rates(param_name, param_choices, success_rates, title=''):
    plt.plot(param_choices, success_rates, '-bo', label='success rate')
    plt.xlabel(param_name)
    title = f'{param_name} success rates' if not title else title
    plt.title(title)
    plt.legend()
    plt.show()


def cross_validate(classifier, training_set, training_labels, num_folds=5, show_graphs=True, **params):
    best_params = {}
    training_size = training_set.shape[0]
    fold_size = int(training_size / num_folds)

    for param_name, choices in params.items():

        success_results = np.zeros((len(choices), num_folds))

        for choice_num, param in enumerate(choices):

            for i in range(num_folds):
                (training_folds, training_folds_labels,
                 validation_fold, validation_fold_labels) = divide_to_folds(training_set, training_labels, fold_size, i)

                classifier.train(training_folds, training_folds_labels)

                result_labels = classifier.predict(validation_fold, **{param_name: param})
                success_rate = get_success_rate(result_labels, validation_fold_labels)
                success_results[choice_num, i] = success_rate

        averaged_success_rates = np.average(success_results, axis=1)
        if show_graphs:
            show_parameter_success_rates(param_name, choices, averaged_success_rates,
                                         title=f'cross-validation of parameter {param_name}')
        max_idx = int(np.argmax(averaged_success_rates))
        max_param = choices[max_idx]
        best_params[param_name] = max_param

    return best_params


# deprecated
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