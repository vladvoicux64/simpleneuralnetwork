# abstract base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and updates parameters if any)

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
