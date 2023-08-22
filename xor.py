import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from tanh import tanh, tanh_derivative
from loss import mse, mse_derivative

training_input = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
training_output = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add_layer(FCLayer(2, 10))
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FCLayer(10, 1))
net.add_layer(ActivationLayer(tanh, tanh_derivative))

net.use_loss(mse, mse_derivative)
net.train(training_input, training_output, epoch_count=100000, learning_rate=0.001)

out = net.network_forward_propagation(training_input)
print(out)
