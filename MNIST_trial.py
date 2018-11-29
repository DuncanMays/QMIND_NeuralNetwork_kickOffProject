from Model import NeuralNet
from TrainingSystem import TrainingSystem

test = TrainingSystem([3,5,10,5,3])

test.backpropegateError([1,2,3], [1,2,3])