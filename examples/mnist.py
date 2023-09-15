import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from src.activation_functions.sigmoid import sigmoid, sigmoid_derivative
from src.layers.activation_layer import ActivationLayer
from src.layers.convolutional_layer import ConvolutionalLayer
from src.layers.fc_layer import FCLayer
from src.layers.reshape_layer import ReshapeLayer
from src.losses.mse import mse, mse_derivative
from src.network import Network

(training_input, training_output), (test_input, test_output) = mnist.load_data()


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 1, 2)
    return x, y


training_input, training_output = preprocess_data(training_input, training_output, 500)
test_input, test_output = preprocess_data(test_input, test_output, 500)

net = Network()
net.layers = [
    ConvolutionalLayer((1, 28, 28), 3, 5, True),
    ActivationLayer(sigmoid, sigmoid_derivative),
    ReshapeLayer((5, 26, 26), (1, 5 * 26 * 26)),
    FCLayer(5 * 26 * 26, 100, True),
    ActivationLayer(sigmoid, sigmoid_derivative),
    FCLayer(100, 2, True),
    ActivationLayer(sigmoid, sigmoid_derivative),
]

net.use_loss(mse, mse_derivative)

net.train_stochastic(training_input, training_output, 20, 0.1)

out = np.round(net.network_forward_propagation(test_input))

correct_ans = 0
for prediction, answer in zip(out, test_output):
    if np.array_equal(prediction, answer): correct_ans += 1

print('accuracy:{}% '.format(i / 10))
