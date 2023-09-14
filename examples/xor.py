import numpy as np

from src.activation_functions.leakyReLU import leaky_relu, leaky_relu_derivative
from src.layers.activation_layer import ActivationLayer
from src.layers.fc_layer import FCLayer
from src.losses.mse import mse, mse_derivative
from src.network import Network

training_input = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
training_output = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.layers = [
    FCLayer(2, 2, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative),
    FCLayer(2, 1, True),
    ActivationLayer(leaky_relu, leaky_relu_derivative)
]

net.use_loss(mse, mse_derivative)
net.train_stochastic(training_input, training_output, epoch_count=1000, learning_rate=0.1)

out = np.round(net.network_forward_propagation(training_input)).reshape(4,1)
print(out)
