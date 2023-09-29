import numpy as np


def accuracy_measure(input, output, net):
    net_output = np.round(net.network_forward_propagation(input))
    correct_ans = 0
    for prediction, answer in zip(net_output, output):
        if np.array_equal(prediction, answer): correct_ans += 1
    return correct_ans / len(output) * 100


def accuracy_statistic(test_input, test_output, training_input, training_output, net):
    test_accuracy = accuracy_measure(test_input, test_output, net)
    validation_accuracy = accuracy_measure(training_input, training_output, net)

    if validation_accuracy < 90 and test_accuracy < 95:
        verdict = 'underfitting'
    elif validation_accuracy > 95 and test_accuracy < 90:
        verdict = 'overfitting'
    else:
        verdict = 'neither underfitting or overfitting'

    print(
        'test_accuracy:{}%   validation_accuracy:{}%   verdict:{}'.format(test_accuracy, validation_accuracy, verdict))
