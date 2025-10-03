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
        if len(self.weights_list) != len(inputs_list):
            raise ValueError(f"weights ({len(self.weights_list)}) != inputs ({len(inputs_list)})")
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
        return self.sigmoidActivation(self.linearTransformation(inputs_list))    

class NetworkLayer():
    def __init__(self, neurons_list, size = 0):
        self.neurons_list = neurons_list #[n1,n2,n3...nn]
        self.size = size

    def inputLayerForwardPass(self, inputs_list):
        '''
        Parses the data through the input layer, ensuring a dataset larger than the input layer is handled correctly
        '''
        return inputs_list[:self.size]

    def layerForwardPass(self, inputs_list):
        '''
        Genereates list of each neuron activation in a layer 
        '''
        neuron_activations = []
        for neuron in self.neurons_list:
            neuron_activations.append(neuron.neuronForwardPass(inputs_list))
        return neuron_activations

class NeuralNetwork():
    def __init__(self, network_layers_list):
        self.network_layers_list = network_layers_list #[l1,l2,l3...ln]
    
    def forwardPass(self, inputs_list):
        '''
        Passes activations from input layer through hidden layer to output layer
        '''
        features = inputs_list[1:]

        #Input layer
        input_layer = self.network_layers_list[0]
        activations_list = input_layer.inputLayerForwardPass(features)

        #Hidden layer(s)
        for hidden_layer in self.network_layers_list[1:-1]:
            activations_list = hidden_layer.layerForwardPass(activations_list)

        #Output layer
        output_layer = self.network_layers_list[-1]
        activations_list = output_layer.layerForwardPass(activations_list)

        return activations_list
    
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
        sys.exit(1)
    
    input_layer_neurons =[]
    for i in range(0, input_neurons, 1):
        input_layer_neurons.append(Neuron())

    return NetworkLayer(input_layer_neurons, size=input_neurons)

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

def generateCorrectLabels(dataset, output_neuron_amount):
    '''
    Generates a 
    '''
    correct_label_list = []
    for datapoint in dataset:
        label = datapoint[0]  # first column is the class label
        one_hot = [0] * output_neuron_amount
        one_hot[label] = 1
        correct_label_list.append(one_hot)

    return correct_label_list

def calculateLogLoss(correct_label_list, label_predictions_list):
        '''
        Returns the entropy loss of a list of datapoints
        '''
        label_predictions_amount = len(label_predictions_list)
        correct_label_list = correct_label_list[:label_predictions_amount]

        log_loss = 0.0

        for i in range(len(correct_label_list)):  # loop over datapoints
            correct_labels = correct_label_list[i]
            predicted_labels = label_predictions_list[i]

            for j in range(len(correct_labels)):  # loop over classes
                if correct_labels[j] == 1:  # only the true class contributes
                    log_loss += -math.log(predicted_labels[j])

        return log_loss / len(correct_label_list)
                

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
    
    prediction_neuron_activations = []
    for datapoint in training_data:
        prediction_neuron_activations.append(network.forwardPass(datapoint))
    print(prediction_neuron_activations)

    correct_label_list = generateCorrectLabels(training_data, NOutput)
    log_loss = calculateLogLoss(correct_label_list, prediction_neuron_activations)
    print(log_loss)
    

if __name__ == "__main__":
    main()


