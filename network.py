class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    # for adding layers
    def add_layer(self, layer):
        self.layers.append(layer)

    # setting up the loss function
    def use_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    # forward prop.
    def network_forward_propagation(self, input_data):
        # get no. of samples
        samples_count = len(input_data)
        result = []

        # actual forward prop.
        for i in range(samples_count):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # training (note: training_output repr. the expected value, the real result)
    def train(self, training_input, training_output, epoch_count, learning_rate):
        sample_count = len(training_input)

        for i in range(epoch_count):
            display_loss = 0
            for j in range(sample_count):
                # forward prop.
                output = training_input[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute error (only for display)
                display_loss += self.loss(training_output[j], output)

                # backward prop.
                loss = self.loss_derivative(training_output[j], output)
                # got to reverse the layers to propagate backwards (duh)
                for layer in reversed(self.layers):
                    loss = layer.backward_propagation(loss, learning_rate)

            # calculate average error
            display_loss /= sample_count
            print('epoch {}/{}  error={}'.format(i+1, epoch_count, display_loss))

