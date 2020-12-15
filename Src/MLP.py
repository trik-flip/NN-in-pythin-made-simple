import numpy as np
from random import random


class MLP:
    """A simple Multi layer network implementation"""

    def __init__(self, num_input=3, num_hidden=[3, 3], num_output=2) -> None:
        """Create a MLP"""
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        layers = [self.num_input] + self.num_hidden + [self.num_output]
        self.weigths = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weigths.append(w)
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, activations):
        self.activations[0] = activations
        """AKA Predict"""
        for i, w in enumerate(self.weigths):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        return activations

    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivatives(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(
                current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(
                current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weigths[i].T)
            if verbose:
                print(f"derivatives for W{i}: {self.derivatives[i]}")
        return error

    def gradient_decent(self, learning_rate, verbose=False):
        for i in range(len(self.weigths)):
            weights = self.weigths[i]
            if verbose:
                print(f"Original W{i}: {weights}")
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            if verbose:
                print(f"Updated  W{i}: {weights}")

    def train(self, inputs, targets, epochs, learning_rate, verbose=2):
        sum_error = 0
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_decent(learning_rate)
                sum_error += self._mse(target, output)

            if verbose > 1:
                print(f"Error {sum_error/len(inputs)} at epoche {i+1}")
        if verbose:
            print(f"Error {sum_error/len(inputs)} at epoche {epochs}")

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivatives(self, x):
        return x*(1 - x)

    def _sigmoid(self, x):
        """Return Sigmoid function of x"""
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    mlps = []
    best = None
    min_error = 1
    for x in range(2, 8):
        mlps.append((x, MLP(2, [x], 1)))
    for i, mlp in mlps:
        mlp.train(inputs, targets, 50, 0.1, verbose=0)
        input = np.array([.1, .2])
        result = mlp.forward_propagate(input)
        error = (0.3 - result)**2
        if error < min_error:
            best = i
            min_error = error
    print(f"the best = {best}, with a error of {min_error}")
