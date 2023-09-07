import sys

import numpy as np

sys.path.append('../src/')

from src.network import Network
from src.layers.fc_layer import FCLayer
from src.layers.activation_layer import ActivationLayer
from src.leakyReLU import leaky_relu, leaky_relu_derivative
from src.losses.mse import mse, mse_derivative

from keras.datasets import mnist
from keras.utils import to_categorical

(training_input, training_output), (test_input, test_output) = mnist.load_data()

training_input = training_input.reshape(training_input.shape[0], 1, 28 * 28)
training_input = training_input.astype('float32')
training_input /= 255

training_output = to_categorical(training_output)

test_input = test_input.reshape(test_input.shape[0], 1, 28 * 28)
test_input = test_input.astype('float32')
test_input /= 255
test_output = to_categorical(test_output)

net = Network()
net.add_layer(FCLayer(28 * 28, 14, True))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))
net.add_layer(FCLayer(14, 10, True))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))

net.use_loss(mse, mse_derivative)

for i in range(60):
    print(f'batch {i}:')
    net.train(training_input[i * 1000: (i + 1) * 1000], training_output[i * 1000: (i + 1) * 1000], epoch_count=100,
              learning_rate=0.1)

out = np.round(net.network_forward_propagation(test_input)).reshape(10000, 10)
print(out)
