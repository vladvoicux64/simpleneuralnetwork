import sys

import numpy as np

sys.path.append('../src/')

from src.network import Network
from src.layers.fc_layer import FCLayer
from src.layers.activation_layer import ActivationLayer
from src.activation_functions.leakyReLU import leaky_relu, leaky_relu_derivative
from src.losses.mse import mse, mse_derivative

training_input = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
training_output = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add_layer(FCLayer(2, 2, False))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))
net.add_layer(FCLayer(2, 1, False))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))

net.use_loss(mse, mse_derivative)
net.train(training_input, training_output, epoch_count=1000, learning_rate=0.1)

out = np.asarray(np.round(net.network_forward_propagation(training_input)))
print(out)
