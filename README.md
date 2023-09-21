# simpleneuralnetwork
This is a simple neural network framework I implemented. This project was intended as an exercise for myself, so feel free to point out any mistakes or to come up with suggestions. I took inspiration from a few articles on [towardsdatascience.com](https://towardsdatascience.com/).

# Features

The framework offers support for fully connected NNs and convolutional NNs, with stochastic and mini-batch training
options. It also comes packed with some popular activations and loss functions.

# Usage

### Prerequisites

Clone the repository and install dependencies in your prefered python environment
using ```pip install -r requirements.txt```

### The network class

The network class features the layer array and 4 methods:

1)```layers = [<insert layers>]```

used to declaratively set layers

2)```use_loss(loss, loss_derivative)```

used to set the loss function of the network;

3)```network_forward_propagation(input)```

used for network inference;

4)```train_stochastic(training_input, training_output, epoch_count, learning_rate)```

used to train using SGD;

5)```train_minibatch(training_input, training_output, epoch_count, batch_size, learning_rate)```

used to train using mini-batch GD;

### Layer types

In the layers folder you will find 4 layer types. Their respective constructors (which you should use to set
the ```layers``` array) are used as follows:

1)```ActivationLayer(activation, activation_derivative)```

this layer applies the activation function to it's input and outputs the result, taking the activation function and it's
derivativce as parameters;

2)```ConvolutionalLayer(input_shape, kernel_size, depth, init_negative_weights=False)```

the convolutional layer which takes as parameters the input shape, the kernel size (side of a square), depth (number of
filters) and a boolean which toggles the initialization of negative weights;

3)```FCLayer(input_size, output_size, init_negative_weights=False)```

the fully connected layer which takes as parameters the input and output sizes (number of neurons), and a boolean which
toggles the initialization of negative weights;

4)```ReshapeLayer(input_shape, output_shape):```

this layer reshapes the input to the output shape and vice-versa;

### Examples

You can find two usage examples in the ```examples``` folder, each employing different types of networks (the MNIST
example even has some simple data preprocessing)

# Performance

The framework does not employ parallelized computing techniques because that came as a second thought to me, meaning I
would have had to rewrite a large part of the project from scratch. I tried to implement that using ```numba``` but that
also did not work well with the OOP structure.
Still, using a reduced number of examples you may train complex models too, with satisfactory results: the XOR model
trains (with a few tries because the staring weights are randomized) to a low error in the realm of 3e-32 with 100%
accuracy. The MNIST model, depending on the training method, may achieve errors in the realm of 6e-08 with accuracy
ranging between 96 - 100% on unseen data.  