import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def use_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def network_forward_propagation(self, input):
        samples_count = len(input)
        result = []

        for i in range(samples_count):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def train_stochastic(self, training_input, training_output, epoch_count, learning_rate):
        sample_count = len(training_input)

        for i in range(epoch_count):
            display_loss = 0
            for j in range(sample_count):
                output = training_input[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                display_loss += self.loss(training_output[j], output)

                loss = self.loss_derivative(training_output[j], output)
                for layer in reversed(self.layers):
                    loss = layer.backward_propagation(loss, learning_rate)

            display_loss /= sample_count
            print('epoch {}/{}  error={}'.format(i + 1, epoch_count, display_loss))

    def train_minibatch(self, training_input, training_output, epoch_count, batch_size, learning_rate):
        input_batches = [training_input[i:i + batch_size] for i in range(0, len(training_input), batch_size)]
        output_batches = [training_output[i:i + batch_size] for i in range(0, len(training_output), batch_size)]
        for i in range(epoch_count):
            display_loss = 0
            for batch_input, batch_output in zip(input_batches, output_batches):
                loss = 0
                output = self.network_forward_propagation(batch_input)
                loss = np.mean(self.loss_derivative(batch_output, output))
                display_loss += np.mean(self.loss(batch_output, output))
                for layer in reversed(self.layers):
                    loss = layer.backward_propagation(loss, learning_rate)

            display_loss /= len(training_input) / batch_size
            print('epoch {}/{}  error={}'.format(i + 1, epoch_count, display_loss))
