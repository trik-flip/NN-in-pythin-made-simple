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
