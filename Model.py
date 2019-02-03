"""This code was written by Duncan Mays for the QMIND Neural Net Design Team.
It is an origional neural network library, similar to tesorflow, the only dependance are the numpy and math libraries
I made this with the intention that it will be used as teaching material, so that new QMIND members can study this code to learn how neural networks work.
You'll notice that this neural network has a preference for being trained with backpropegation, this is because backpropegation is the first thing i will
use it with. I will hopefully get around to fixing this, if not completely rewriting it to be more versatile."""

import numpy as np 
from math import exp

class NeuralNet:
	
	#since there multiple init functions where each of these varibles are properly instantiated, and understanding these variabes is important so
	#I'll be quite extensive, I've chosen to describe the three variables that describe the network right now and not where they're instantiated:

	# ca = []
	#ca stands for characteristic array
	#each index holds the number of neurons in the layer corresponding to that index, note that the input and output vectors are considered layers
	#for example, a ca of [3,2,5,4] represents a net that has 3 inputs, a hidden layer with 2 neurons, a hidden layer with 5 neurons and 4 outputs
	#note that most neural networks do not assign biases to the output layer, this was a design choice on my part.

	# biases = []
	#this array holds the biases for each neuron in the net
	#note that this is a normal python array of numpy arrays
	#the first index repsresents the layer being refered to, so it's length will be the length of ca-1, since ca counts inputs as a layer
	#the second index represents the specific neuron whos bias is stored at that index combination, so the second dimension will have the 
	#length of the value in the first index in ca, since that is the number of neurons in that layer

	# weights = []
	#this will be an array of matrices that represents the weighted connections between each neuron in the network
	#this is an array of 2D numpy arrays
	#there will be len(ca)-1 elements in this array, since there are weighted connections between the layers, not for each layer
	#the number of columns in a matrix equals the number of neurons in the layer before it, and the number of rows the number of neurons in the layer after
	#so the first dimention of each numpy array would be ca[i] and the second dimension ca[i+1] in an iterative loop

	# DNA = [ca, biases, weights]
	#DNA is one array that holds all the information describing the net, it literally just is ca, biases, and weights shoved together into an array
	#The order of elements is very important for the array to be usefully referenced later, so make a mental note that ca is in index 0, biases 1, and wieghts 2
	#this array will be used to pass information about the neural net to parent objects and for parent objects to alter the network without having to boot up
	#a new instance

	#this will be the activation function for the network, each neuron will pass the weighted sum plus bias of its inputs into this function, and the pump the
	#output of this function on to the next layer. The prime version of the activation is it's derivative, and is necessary for backpropegation
	"""TODO includ functionality other than sigmoid"""
	sigma = lambda self, x : 2*(exp(x)/(1+exp(x)))-1
	sigmaPrime = lambda self, x : 2*exp(x)/((1+exp(x))**2)

	#this is the default constructor, it will initiate the features of the net (biases and weights) randmly according to a given ca
	"""TODO throw exceptions when input is invalid"""
	def __init__(self, inCA):

		#the three variables that describe the neural network, they must be instantiated within the init method so that they can belong to each instance
		#individually, rather than being shared among all instances. Before the init method there is no instance to attach them to, so they're just kinda
		#static.
		self.ca = inCA
		self.biases = []
		self.weights = []

		#iteratively instantiates biases and weigths with random values
		for i in range(0, len(inCA)-1):
			#numpy.random.rand() returns an array of the given shape of random numbers between 0 and 1, so 2*np.random.rand()-1 returns between -1 and 1
			self.biases.append(2*np.random.rand(inCA[i+1])-1)
			#the weights will be set between -3 and 3
			self.weights.append(6*np.random.rand(inCA[i+1], inCA[i])-3)

		#just a package for all the information that describes the network
		self.DNA = [self.ca, self.biases, self.weights]

	#this function takes in a new DNA strand and sets all the weights and biases, and importantly ca, according to it
	#it basically recreates the Model instance with a new neural network, this is so any parent object doesn't have to reintantiate the class when it needs a new one
	#the input is just a new DNA strand
	def mutate(self, inDNA):
		self.DNA = inDNA
		self.ca = inDNA[0]
		self.biases = inDNA[1]
		self.weights = inDNA[2]

	#basically the same function as run, takes an input vector and feeds it forward in the network. The difference here is that this function will save the 
	#activations at each layer, and the return them in the form of an array. This is needed for backpropegation. 
	def getWeightedSums(self, input):

		#input counts as an activation
		sums = [input]

		#this block iteratively multiplies the vector outputed from the last layer in the network, adds the bias and puts it through the activation function
		#before repeating the process for the next layer using the variable temp to store the vector between steps
		temp = self.weights[0].dot(input) + self.biases[0]
		sums.append(temp)
		temp = self.iterate(self.sigma, temp)
		for i in range(1, len(self.ca)-1):
			temp = self.weights[i].dot(temp) + self.biases[i]
			sums.append(temp)
			temp = self.iterate(self.sigma, temp)

		return sums

	#very self-explanatory
	def getDNA(self):
		return self.DNA

	#this function takes a lambda function and an array, and returns and array with each element being the value of the lambda function given the corresponding
	#value in the input array
	def iterate(self, function, input):
		A = [function(input[0])]
		for i in range(1, len(input)):
			A.append(function(input[i]))

		return A

	#this function takes an input vector, plugs it into the neural network and returns whatever comes out of the network
	#on a lower level it takes the input vector, multiplies it by the weight matrix, adds the bias and then repeats for each layer in the network in a recursive 
	#proccess called forward propegation.
	def run(self, input):

		#this block iteratively multiplies the vector outputed from the last layer in the network, adds the bias and puts it through the activation function
		#before repeating the process for the next layer using the variable temp to store the vector between steps
		#do not expect this to be easy to understand
		temp = self.iterate(self.sigma, self.weights[0].dot(input) + self.biases[0])
		for i in range(1, len(self.ca)-1):
			temp = self.iterate(self.sigma, self.weights[i].dot(temp) + self.biases[i])

		return temp


