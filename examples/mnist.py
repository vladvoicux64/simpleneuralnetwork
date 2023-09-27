import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from src.accuracy_statistic import accuracy_statistic
from src.activation_functions.leaky_relu import leaky_relu, leaky_relu_derivative
from src.activation_functions.softmax import softmax, softmax_derivative
from src.layers.activation_layer import ActivationLayer
from src.layers.convolutional_layer import ConvolutionalLayer
from src.layers.fc_layer import FCLayer
from src.layers.maxpooling_layer import MaxpoolingLayer
from src.layers.reshape_layer import ReshapeLayer
from src.losses.xentropy import xentropy, xentropy_derivative
from src.network import Network


def preprocess_data(input, output, count):
    one_indexes = np.where(output == 1)[0][:count]
    zero_indexes = np.where(output == 0)[0][:count]
    mixed_indexes = np.hstack((one_indexes, zero_indexes))
    mixed_indexes = np.random.permutation(mixed_indexes)
    input = input[mixed_indexes]
    output = output[mixed_indexes]
    input = (input.reshape(len(input), 1, 28, 28)).astype("float32") / 255
    output = to_categorical(output)
    output = output.reshape(len(output), 1, 2)
    return input, output


(training_input, training_output), (test_input, test_output) = mnist.load_data()

training_input, training_output = preprocess_data(training_input, training_output, 500)
test_input, test_output = preprocess_data(test_input, test_output, 50)


net = Network()
net.layers = [
    ConvolutionalLayer((1, 28, 28), 3, 32, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative),
    MaxpoolingLayer((32, 26, 26), (2, 2)),
    ConvolutionalLayer((32, 13, 13), 3, 16, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative),
    MaxpoolingLayer((16, 11, 11), (2, 2)),
    ConvolutionalLayer((16, 5, 5), 3, 16, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative),
    ReshapeLayer((16, 3, 3), (1, 144)),
    FCLayer(144, 100, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative),
    FCLayer(100, 2, True),
    ActivationLayer(softmax, softmax_derivative),
]

net.use_loss(xentropy, xentropy_derivative)

net.train_minibatch(training_input, training_output, epoch_count=20, batch_size=1, learning_rate=0.00004)

out = np.round(net.network_forward_propagation(test_input))

accuracy_statistic(test_input, test_output, training_input, training_output, net)
