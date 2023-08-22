import numpy as np
import sys
sys.path.append('../src/')

from src.network import Network
from src.fc_layer import FCLayer
from src.activation_layer import ActivationLayer
from src.leakyReLU import leaky_relu, leaky_relu_derivative
from src.loss import mse, mse_derivative

training_input = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
training_output = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add_layer(FCLayer(2, 2, True))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))
net.add_layer(FCLayer(2, 1, True))
net.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))

net.use_loss(mse, mse_derivative)
net.train(training_input, training_output, epoch_count=1000, learning_rate=0.1)

out = net.network_forward_propagation(training_input)
print(out)
