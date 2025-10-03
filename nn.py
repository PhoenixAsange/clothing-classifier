import math
import random
import sys
import gzip
import csv

class Neuron():
    '''
    Class that handles the neruons function in the neural network
    '''
    def __init__(self, weights_list = None, bias = 0.0):
        self.weights_list = weights_list #[w1,w2,w3...wn]
        self.bias = bias

    def linearTransformation(self, inputs_list):
        '''
        Calculates the linear transformation of the inputs:
        z = (w1 * x1 + w2 * x2 + ... + wn * xn) + bias
        '''
        sum = 0
        for i in range(len(self.weights_list)):
            sum += self.weights_list[i] * inputs_list[i]
        return sum + self.bias
    
    def sigmoidActivation(self, z):
        '''
        Calculates the sigmoid activation of z.
        Formula: σ(z) = 1 / (1 + e^(-z))
        '''
        return 1 / (1 + math.exp(-z))
    
    def neuronForwardPass(self, inputs_list):
        '''
        Procceses an input through a neuron
        '''
        z = self.linearTransformation(inputs_list)
        return self.sigmoidActivation(z)    

class NetworkLayer():
    def __init__(self, neurons_list):
        self.neurons_list = neurons_list #[n1,n2,n3...nn]

    def layerForwardPass(self, inputs_list):
        '''
        Genereates list of each neuron activation in a layer 
        '''
        outputs_list = []
        for neuron in self.neurons_list:
            outputs_list.append(neuron.neuronForwardPass(inputs_list))
        return outputs_list

class NeuralNetwork():
    def __init__(self, network_layers_list):
        self.network_layers_list = network_layers_list #[l1,l2,l3...ln]
    
    def forwardPass(self, inputs_list):
        '''
        Passes activations from start layer to the last layer
        '''
        outputs_list = inputs_list
        for layer in self.network_layers_list:
            outputs_list = layer.layerForwardPass(outputs_list)
        return outputs_list
    
def loadDataset(filename):
    """
    Load CSV.gz dataset into a list of rows
    """
    dataset = []
    with gzip.open(filename, 'rt') as file:
        reader = csv.reader(file)
        next(reader) # skip labels
        for row in reader:
            datapoint = []
            for value in row:
                datapoint.append(int(value))
            dataset.append(datapoint)
    return dataset
    
def generateInputLayer(input_neurons, dataset):
    '''
    Generates the input layer with a variable amount of input neurons
    '''
    if len(dataset[0]) - 1 < input_neurons:
        print(f"Not enough training data for {input_neurons} neurons")
        exit()
    
    input_layer_neurons =[]
    for i in range(0, input_neurons, 1):
        input_layer_neurons.append(Neuron())
    return NetworkLayer(input_layer_neurons)

def generateRandomWeight(lower_bound, upper_bound):
    '''
    Generates a random weight given a range
    '''
    return random.uniform(lower_bound, upper_bound)


def generateHiddenLayer(hidden_neuron_amount, input_layer_neuron_amount):
    '''
    Generates the hidden layer with a variable amount of hidden neurons
    '''
    hidden_layer_neurons =[]
    for i in range(0, hidden_neuron_amount, 1):
        weight_list = []
        for i in range(input_layer_neuron_amount):
            weight_list.append(generateRandomWeight(-0.01, 0.01))
        hidden_layer_neurons.append(Neuron(weight_list, 0.0))
    return NetworkLayer(hidden_layer_neurons)

def generateOutputLayer(output_neuron_amount, hidden_layer_neuron_amount):
    '''
    Generates the output layer with a variable amount of output neurons
    '''
    output_layer_neurons =[]
    for i in range(0, output_neuron_amount, 1):
        weight_list = []
        for i in range(hidden_layer_neuron_amount):
            weight_list.append(generateRandomWeight(-0.01, 0.01))
        output_layer_neurons.append(Neuron(weight_list, 0.0))
    return NetworkLayer(output_layer_neurons)

def main():
    if len(sys.argv) != 6:
        print("Usage: python nn.py NInput NHidden NOutput train.csv.gz test.csv.gz")
        sys.exit(1)

    # Parse command-line arguments
    NInput = int(sys.argv[1])
    NHidden = int(sys.argv[2])
    NOutput = int(sys.argv[3])
    train_file = sys.argv[4]
    test_file = sys.argv[5]

    print(f"Building network with {NInput} input, {NHidden} hidden, {NOutput} output neurons")

    # Load datasets
    training_data = loadDataset(train_file)
    testing_data = loadDataset(test_file)

    print(f"Loaded {len(training_data)} training examples")
    print(f"Loaded {len(testing_data)} test examples")

    network = NeuralNetwork([generateInputLayer(NInput, training_data), 
                             generateHiddenLayer(NHidden, NInput), 
                             generateOutputLayer(NOutput, NHidden)])

    

if __name__ == "__main__":
    main()


