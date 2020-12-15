import unittest
from random import random

import numpy as np
from Src.MLP import MLP

INPUTS = np.array([[random()/2 for _ in range(2)]
                   for _ in range(1000)])
TARGETS = np.array([[i[0] + i[1]] for i in INPUTS])
MLP_2 = MLP(2, [3], 1)
MLP_2.train(INPUTS, TARGETS, 50, 0.1, verbose=0)


class MainTests(unittest.TestCase):
    def test_create_MLP_Model(self):
        MLP_2 = MLP(1, [1, 1, 1], 1)
        self.assertEqual(len(MLP_2.weigths), 1 + len(MLP_2.num_hidden))

    def test_predict_small_numbers(self):
        INPUT = np.array([.00001, .00009])
        OUTPUT = .0001
        self.assertAlmostEqual(
            MLP_2.predict(INPUT), OUTPUT)

    def test_predict_normal_numbers(self):
        INPUT = np.array([.1, .2])
        OUTPUT = .3
        self.assertAlmostEqual(
            MLP_2.predict(INPUT), OUTPUT, places=1)

    def test_predict_zeros(self):
        INPUT = np.array([0, 0])
        OUTPUT = 0
        self.assertAlmostEqual(
            MLP_2.predict(INPUT), OUTPUT, places=1)

    def test_predict_3_numbers(self):
        INPUTS = np.array([[random()/2 for _ in range(3)]
                           for _ in range(1000)])
        TARGETS = np.array([[i[0] + i[1] + i[2]] for i in INPUTS])
        MLP_3 = MLP(3, [4], 1)
        MLP_3.train(INPUTS, TARGETS, 100, 0.1, verbose=0)
        INPUT = np.array([.3, .3, .1])
        OUTPUT = .7
        self.assertAlmostEqual(
            MLP_3.predict(INPUT), OUTPUT, places=1)


if __name__ == "__main__":
    unittest.main()
