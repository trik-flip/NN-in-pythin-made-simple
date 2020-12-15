import numpy as np


class MLP:

    def __init__(self, num_input=3, num_hidden=[3, 5], num_output=2) -> None:
        self.num_input = [num_input]
        self.num_hidden = num_hidden
        self.num_output = [num_output]

        layers = self.num_input + self.num_hidden + self.num_output

        self.weigths = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weigths.append(w)

    def forward_propagate(self, activations):
        for w in self.weigths:
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    mlp = MLP()

    inputs = np.random.rand(mlp.num_input)


    ouputs = mlp.forward_propagate(inputs)

    print(f"input:{inputs}")
    print(f"output:{ouputs}")
