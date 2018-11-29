from Model import NeuralNet
import numpy as np

class TrainingSystem:

	#this is the model that this system will train
	model = NeuralNet([])

	#initializes the system with a neural net of characteristic array given by parent program
	def __init__(self, inCA):
		self.model = NeuralNet(inCA)

	#this functions takes the input to the model and the model's desired out put
	#its outputs an array of numpy arrays containing the error in each layer of the model
	#you will not understand this
	def backpropegateError(self, input, desiredOutput):

		#the wieghted sums of each neuron in the network, needed for backpropegation
		weightedSums = self.model.getWeightedSums(input)

		#the output of the model
		#the last layer's weighted sums put through the self.model's activation functions
		actualOutput = self.model.iterate(self.model.sigma, weightedSums[len(weightedSums)-1])

		modelDNA = self.model.getDNA()
		modelCA = modelDNA[0]
		modelBiases = modelDNA[1]
		modelWeights = modelDNA[2]

		#this array will hold the errors in each layer and will be constricted backwards
		#the first element we will add is the error in the last layer:
		bwrdsError = [np.multiply(np.array(self.costDerivative(actualOutput, desiredOutput)), (self.model.iterate(self.model.sigmaPrime, weightedSums[len(weightedSums)-1])))]

		for i in range(0, len(modelCA)-1):

			#[w^(l+1)]^T = transpose of the weight matrix to the next layer = modelWeights[len(modelWeights)-1-i].transpose()
			#delta^(l+1) = error of the next layer = bwrdsError[len(bwrdsError)-1]
			#sigmaPrime(z^l) = rate of change of the activation function at its current activation = self.model.iterate(self.model.sigmaPrime, weightedSums[len(weightedSums)-1-i])

			#this calculation is complicated, so i'm gunna split it up
			firstFactor = np.multiply(modelWeights[len(modelWeights)-1-i].transpose(), bwrdsError[len(bwrdsError)-1])

			secondFactor = self.model.iterate(self.model.sigmaPrime, weightedSums[len(weightedSums)-1-i])

			bwrdsError.append(np.multiply(firstFactor, secondFactor))


	#cost funtion for the neural network, returns half the sum of the squares of the difference between the actual output of the network, and the desired output.
	#I chose this because it's derivative is just the difference between the output of the network and the desired output.
	def cost(self, actualOutput, desiredOutput):
		sum = 0

		for i in range(0, len(actualOutput)):
			sum += (actualOutput[i] - desiredOutput[i])**2

		return sum/2

	#This function can somewhat be thought of as the derivative of the cost function.
	#Since the cost function is quadratic, its derivative relative to each element in the output vector is just the difference between that element and 
	#the corresponding element in the desired output vector. This function, takes the actual output and desired output vectors, and returns their component-
	#wise differenct, effectively returning a vector of derivatives of the cost function, with each element relative to the corresponding element in the 
	#actual output vector.
	def costDerivative(self, actualOutput, desiredOutput):
		A = []

		for i in range(0, len(actualOutput)):
			A.append(actualOutput[i] - desiredOutput[i])

		return A

	#this function returns the model that this system has been training
	def getModel(self):
		return model

	#takes a characteristic array from a parent program and sets the system up to train a new neural net of that characteristic array
	def newModel(self, inCA):
		model = NeuralNet(inCA)

	#if the system needs to be used to train a different model, this function will recieve it from the parent program
	def setModel(self, inModel):
		model = inModel
