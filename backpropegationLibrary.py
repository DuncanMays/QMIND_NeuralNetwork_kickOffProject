import numpy as np

#this function alters the DNA of neural net according to a gradient provided by the above getGradient function
#it just subtracts the gradient of the network from its wieghts and biases after multiplying by a certain learning rate.
#its inputs are the neural net to be altered, its gratient, and the learning rate.
def alterNet(learningRate, gradient, model):
	#these are the derivatives of the cost function veruss the biases and wieghts respectively
	costVSbiases = gradient[0]
	costVSweights = gradient[1]

	#the aliases of the weights and biases in the network
	biases = model.biases
	weights = model.weights

	for i in range(0, len(biases)):
		biases[i] -= learningRate*costVSbiases[i]
		weights[i] -= learningRate*costVSweights[i]

	

#this function allows for the measurement of the performance of the network
#it returns the sum of the squares of the differences between the individual elements of the desired output and actual output
def cost(actualOutput, correctOutput):
	cost = 0
	for i in range(0, len(actualOutput)):
		cost += (actualOutput[i] - correctOutput[i])**2
	return cost

#this function is the heart of this library, it does all the backpropegating that this library gets its name from
#this function does not alter the neural network it takes as inTake, it calculates the gradient of that network
#using the single trail given to it, in the form of an inTake and an output array
#the inTake array is the inTake vector to the neural net, wheras the output vector is the "correct" output of the 
#neural net. IE it is what the actual output gets compared to in order for backpropegation to take place.
#it has 3 inTakes, the inTake and outputs arrays, and then the model that needs training.
#note that i had to use inTake instead of input since input is a reserved word in python.
def getGradient(inTake, output, model):

	#this is the characteristic array of the model, needed for all sorts of things
	ca = model.ca

	#this is the initialization of the 2D array that represents the derivative of the cost versus the activations in each neuron in each layer
	costVSactivations = []
	for j in range(0, len(ca)):
		costVSactivations.append(np.zeros(ca[j]))

	#this line calculates the gradient relative to the activations in the last layer of the network, hugely important
	#as the rest of the program is build off of this
	costVSactivations[len(costVSactivations)-1] =  2*(np.array(model.run(inTake)) - output)

	#now comes the hard part
	#we need to backpropegate the derivative, ie, use the derivative of cost versus the activations in a layer to find the
	#derivative of cost versus the activations the the layer before it.
		
	#this array represents the weighted sums of all the neurons in the network
	#its shape should correspond to ca
	weightedSums = model.getWeightedSums(inTake)
	#the [:] is to prevent aliasing
	weights = model.weights[:]

	#iterates backwards through the network, you'll notice it starts at the second last index, since we already calculated
	#the derivative versus the activations in the last layer.
	#this is the backpropegation that the process gets its name from.
	for j in range(1, len(ca)):
		#index of the layer in consideration
		k = len(ca) - j

		#this is a 1D array that represents the derivative of the activation function
		actDeriv = np.array(model.iterate(model.sigmaPrime, weightedSums[k]))

		costVSactivations[k-1] = (weights[k-1].transpose()).dot(actDeriv*costVSactivations[k])

	#now we need to use the gradient versus the activations to calculate the gradient versus all the weights and biases

	#this array represents the derivative of the cost function versus the biases in each neuron, instead of generating a 
	#whole new array I'm just gunna use the actual biases array and plug new values in
	#the [:] is to prevent aliasing
	costVSbiases = model.biases[:]

	#the derivative of the cost versus the bias is just sigmaPrime times costVSactivation
	#note that the NeuralNet class in Model.py does not give the inTake layer a bias, so the length of the biases array is
	#one less than CA
	for j in range(1,len(ca)):
		actDeriv = np.array(model.iterate(model.sigmaPrime, weightedSums[j]))
		costVSbiases[j-1] = actDeriv*costVSactivations[j]

	#now to calculate the gradient versus all the weights
	#same idea as with the biases, the array already exists with the right shape, so why redefine it
	#the [:] is to prevent aliasing
	costVSweights = model.weights[:]

	for j in range(0, len(ca)-1):
		actDeriv = np.array(model.iterate(model.sigmaPrime, weightedSums[j+1]))
		activation =  np.array(model.iterate(model.sigma, weightedSums[j]))
		temp = activation*costVSactivations[j]

		costVSweights[j] = np.outer(actDeriv, temp)

	#returns the derivative verus the cost and versus the biases as an array
	return [costVSbiases, costVSweights]




