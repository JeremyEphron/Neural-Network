from numpy import exp, array, random, dot

class Neuron():
	def __init__(self, number_of_synapses):
		# Inital random starting weights for the Neuron's synapses
		self.synaptic_weights = 2 * random.random((number_of_synapses)) - 1

	def __repr__(self):
		return "{}".format(self.synaptic_weights)


class NeuralLayer():
	def __init__(self, neural_density, inputs_per_neuron):
		# Creates a matrix containing all the synaptic weights of Neurons in the layers
		self.neurons = array([Neuron(inputs_per_neuron).synaptic_weights for i in range(neural_density)]).T

	def __repr__(self):
		return "{}".format(self.neurons)

	# The Sigmoid function, which describes an S shaped curve. We pass the weighted sum of the inputs through this function to normalize them between 0 and 1.
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# Calls the __sigmoid() function on the dot product of the inputs and the matrix of synaptic weights.
	def process(self, inputs):
		return self.__sigmoid(dot(inputs, self.neurons))


class NeuralNetwork():
	"""Where the magic happens"""

	def __init__(self, layers):
		self.layers = array(layers)

	def __repr__(self):
		return "{}".format(self.layers)

	# Prints info about the Neural Network
	def info(self):
		for i in range(len(self.layers)):
			plural = 's'

			if len(self.layers[i].neurons) == 1:
				plural = ""

			print("	Layer {}: {} Neuron{} with {} inputs:".format(i, len(self.layers[i].neurons[0]), plural, len(self.layers[i].neurons)))
			print(self.layers[i].neurons)

	# The Neural Network tries to come up with an answer
	def think(self, inputs):
		outputs_arr = [inputs]

		for layer in self.layers:
			outputs_arr.append(layer.process(outputs_arr[-1]))

		return outputs_arr


    # The derivative of the Sigmoid function. This is the gradient of the Sigmoid curve and it indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# Trains the Neural Network by "thinking" of an answer and then adjusting the synaptic weights in accordance with the desired answer
	def train(self, training_set_inputs, training_set_outputs, training_iterations):
		for iteration in range(training_iterations):

			# Pass training set through our Neural Network
			network_outputs = self.think(training_set_inputs)[1:]

			# Calculate the error for each layer and adjust accordingly
			for i in range(len(self.layers)-1, -1, -1):
				if i == len(self.layers)-1:
					error = training_set_outputs - network_outputs[i]
					delta = error * self.__sigmoid_derivative(network_outputs[i])
				else:
					error = dot(delta, self.layers[i+1].neurons.T)
					delta = error * self.__sigmoid_derivative(network_outputs[i])
				if i == 0:
					adjustment = dot(training_set_inputs.T, delta)
				else:
					adjustment = dot(network_outputs[i-1].T, delta)
				self.layers[i].neurons += adjustment

if __name__ == '__main__':
	
	# Seed the random number generator to get the same random numbers each time
	random.seed(1)

	# Create Neural Layers here (with number of neurons and number of inputs per neuron as params)
	layers = [NeuralLayer(5,3), NeuralLayer(1, 5)]

	# Create the Neural Network from the layers
	neural_network = NeuralNetwork(layers)

	# Create the training sets here
	training_set_inputs  = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

	# Let's see what the inital synaptic weights look like
	print("1) Initial Random Synaptic Weights: ")
	neural_network.info()

	# Now train that network
	neural_network.train(training_set_inputs, training_set_outputs, 60000)

	# Let's see how buff these synaptic weights got while training
	print("\n 2) Post-Training Synaptic Weights: ")
	neural_network.info()

	# Let's see if the network can put it's training to use on a new situation
	print("\n 3) Solve a New Problem, [1, 0, 1] -> ?: ")
	outputs = neural_network.think(array([1,0,1]))
	print(outputs[-1])