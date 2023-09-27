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
        start_err = 0
        end_err = 0
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
            if i == 0: start_err = display_loss
            if i == epoch_count - 1: end_err = display_loss
            print('epoch {}/{}  error={}'.format(i + 1, epoch_count, display_loss))

        improvement = (start_err - end_err) / abs(start_err) * 100
        print('rate of improvement={}%'.format(improvement))

    def train_minibatch(self, training_input, training_output, epoch_count, batch_size, learning_rate):
        start_err = 0
        end_err = 0
        input_batches = [training_input[i:i + batch_size] for i in range(0, len(training_input), batch_size)]
        output_batches = [training_output[i:i + batch_size] for i in range(0, len(training_output), batch_size)]
        batch_count = len(training_input) // batch_size

        for i in range(epoch_count):
            display_loss = 0
            for mini_input, mini_output in zip(input_batches, output_batches):
                output = self.network_forward_propagation(mini_input)
                display_loss += np.sum(self.loss(mini_output, output)) / batch_size
                loss = sum(self.loss_derivative(mini_output, output)) / batch_size
                for layer in reversed(self.layers):
                    loss = layer.backward_propagation(loss, learning_rate)

            display_loss /= batch_count
            if i == 0:
                start_err = display_loss
            if i == epoch_count - 1:
                end_err = display_loss
            print('epoch {}/{}  error={}'.format(i + 1, epoch_count, display_loss))

        improvement = (start_err - end_err) / abs(start_err) * 100
        print('rate of improvement={}%'.format(improvement))
