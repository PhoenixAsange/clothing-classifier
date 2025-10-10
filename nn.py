import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import gzip

class Neuron():
    '''
    Class that handles the neurons function in the neural network
    '''
    def __init__(self, weights_list = None, bias = 0.0):
        self.weights_list = weights_list #[w1,w2,w3...wn]
        self.weight_gradients = [0.0] * len(weights_list) if weights_list is not None else [] # Each value corresponds to a weigh in the weight list
        self.bias = bias
        self.activation = 0
        self.delta = 0
        self.bias_gradients = 0

    def linearTransformation(self, inputs_list):
        '''
        Calculates the linear transformation of the inputs.
        Formula: z = (w1 * x1 + w2 * x2 + ... + wn * xn) + bias
        '''
        #Validate input amount corresponds to weight amount
        if len(self.weights_list) != len(inputs_list):
            raise ValueError(f"weights ({len(self.weights_list)}) != inputs ({len(inputs_list)})")
        
        #Compute weighted sum
        total = 0
        for i in range(len(self.weights_list)):
            total += self.weights_list[i] * inputs_list[i]

        #Factor bias
        self.activation = total + self.bias
        return self.activation
    
    def sigmoidActivation(self, z):
        '''
        Calculates the sigmoid activation of the linear transformation.
        Formula: σ(z) = 1 / (1 + e^(-z))
        '''
        #Prevent overflow
        if z >= 0:
            # For positive z: σ(z) = 1 / (1 + e^(-z))
            exp_neg = math.exp(-z)
            self.activation_output = 1.0 / (1.0 + exp_neg)
        else:
            # For negative z: σ(z) = e^z / (1 + e^z) - avoids large e^(-z)
            exp_pos = math.exp(z)
            self.activation_output = exp_pos / (1.0 + exp_pos)
        return self.activation_output
    
    def neuronForwardPass(self, inputs_list):
        '''
        Procceses an input through a neuron
        '''
        return self.sigmoidActivation(self.linearTransformation(inputs_list))    

class NetworkLayer():
    '''
    Class that handles layers of a neural network
    '''
    def __init__(self, neurons_list, size = 0):
        self.neurons_list = neurons_list #[n1,n2,n3...nn]

    def inputLayerForwardPass(self, inputs_list):
        '''
        Passes input through the input neurons, 
        ensuring input and layer size mismatches are handled correctly and normalising pixel data
        '''
        # Handle input/neuron mismatches
        self.size = len(self.neurons_list)
        input_size = len(inputs_list)
        if input_size > self.size:
            # More inputs than neurons - truncate
            processed_inputs = inputs_list[:self.size]
        elif input_size < self.size:
            # Fewer inputs than neurons - pad with zeros
            processed_inputs = inputs_list + [0] * (self.size - input_size)
        else:
            processed_inputs = inputs_list
    
        # Normalise the raw pixel values to 0 or 1
        normalized_inputs = [pixel_value / 255.0 for pixel_value in processed_inputs]
        
        return normalized_inputs

    def layerForwardPass(self, inputs_list):
        '''
        Passes inputs through the layer, returning a list of the activations of the layer neurons
        '''
        neuron_activations = []
        for neuron in self.neurons_list:
            neuron_activations.append(neuron.neuronForwardPass(inputs_list))
        return neuron_activations

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers #[l1,l2,l3...ln]
        self.batch_number = 0
    
    def forwardPass(self, inputs_list):
        '''
        Passes activations from input layer through hidden layer to output layer,
        returning neuron activation values.
        '''
        #Slice label
        input_features = inputs_list[1:]
        
        # Store activations for each layer for backpropagation
        self.layer_activations = []

        # Input layer: preprocess and normalize input data
        self.input_layer = self.layers[0]
        input_activations = self.input_layer.inputLayerForwardPass(input_features)
        self.layer_activations.append(input_activations)   # Store for gradient calculations
        current_activations = input_activations

        # Hidden layer(s): transform inputs through non-linear activation
        for hidden_layer in self.layers[1:-1]:
            self.hidden_layer = hidden_layer
            hidden_activations = hidden_layer.layerForwardPass(current_activations)
            self.layer_activations.append(hidden_activations)  # Store hidden outputs
            current_activations = hidden_activations # Pass to next layer

        # Output layer: produce final predictions
        self.output_layer = self.layers[-1]
        output_activations = self.output_layer.layerForwardPass(current_activations)
        self.layer_activations.append(output_activations)  # Store final outputs

        return output_activations
    
    def computeGradients(self, outputs_activations_list, correct_label):
        '''
        Updates the deltas for every neuron in the network
        
        Formulas:
        Output layer: δ_j = a_j - t_j
        Hidden layer: δ_j = a_j(1-a_j) * sum_over_k(w_jk * δ_k)
        '''
        # Output layer gradients: direct error comparison
        for j in range(len(self.output_layer.neurons_list)):
            # δ_j = a_j - t_j (prediction - target)
            neuron = self.output_layer.neurons_list[j]
            neuron.delta = outputs_activations_list[j] - correct_label[j]

        # Hidden layer gradients: chain rule through downstream neurons
        for j in range(len(self.hidden_layer.neurons_list)):
            # δ_j = a_j(1-a_j) * sum_over_k(w_jk * δ_k)
            # Error from all output neurons flows back to each hidden neuron
            neuron = self.hidden_layer.neurons_list[j]
            activation = neuron.activation_output

            # Sum weighted errors from all connected output neurons: sum_over_k(w_jk * δ_k)
            downstream_sum = 0.0
            for k in range(len(self.output_layer.neurons_list)):
                output_neuron = self.output_layer.neurons_list[k]
                weight_jk = output_neuron.weights_list[j] # Weight from hidden j to output k
                downstream_sum += weight_jk * output_neuron.delta
            
            # Apply sigmoid derivative: a_j(1-a_j) and multiply by downstream error
            neuron.delta = activation * (1.0 - activation) * downstream_sum

    def updateWeights(self, batch_size, learning_rate):
        '''
        Updates weights and biases using accumulated gradients
        
        Formulas:
        Weight update: w_ij = w_ij - η * (1/m) * Σ(δ_i * a_j)
        Bias update: b_i = b_i - η * (1/m) * Σ(δ_i)
        '''        
        # Output layer updates
        for output_neuron in self.output_layer.neurons_list:
            for j in range(len(output_neuron.weights_list)):
                # w_j = w_j - η * (1/m) * Σ(δ * a_prev)
                output_neuron.weights_list[j] -= learning_rate * (output_neuron.weight_gradients[j] / batch_size)
                output_neuron.weight_gradients[j] = 0 # Reset accumulator for next batch
            # b = b - η * (1/m) * Σ(δ)
            output_neuron.bias -= learning_rate * (output_neuron.bias_gradients / batch_size)
            output_neuron.bias_gradients = 0

        # Hidden layer updates  
        for hidden_neuron in self.hidden_layer.neurons_list:
            for i in range(len(hidden_neuron.weights_list)):
                # w_i = w_i - η * (1/m) * Σ(δ * a_prev)
                hidden_neuron.weights_list[i] -= learning_rate * (hidden_neuron.weight_gradients[i] / batch_size)
                hidden_neuron.weight_gradients[i] = 0 # Reset accumulator for next batch
            # b = b - η * (1/m) * Σ(δ)
            hidden_neuron.bias -= learning_rate * (hidden_neuron.bias_gradients / batch_size)
            hidden_neuron.bias_gradients = 0  # Reset accumulator for next batch
    
    def backPropagate(self, outputs_activations_list, correct_label, learning_rate, batch_size):
        '''
        Performs backpropagation for one training example
        '''
        #Compute error gradients for all neurons
        self.computeGradients(outputs_activations_list, correct_label)

        #Accumulate weight gradients for output layer
        for k in range(len(self.output_layer.neurons_list)):
            #weight_gradient += δ_output * a_hidden
            output_neuron = self.output_layer.neurons_list[k]
            for j in range(len(self.hidden_layer.neurons_list)):
                hidden_neuron = self.hidden_layer.neurons_list[j]
                hidden_activation = hidden_neuron.activation_output
                output_neuron.weight_gradients[j] += output_neuron.delta * hidden_activation
            # Accumulate bias gradient
            output_neuron.bias_gradients += output_neuron.delta

        #Accumulate weight gradients for hidden layer
        for j in range(len(self.hidden_layer.neurons_list)):
            #weight_gradient += δ_hidden * a_input
            hidden_neuron = self.hidden_layer.neurons_list[j]
            # weights - use stored input layer activations
            for i in range(len(hidden_neuron.weights_list)):
                input_activation = self.layer_activations[0][i]
                hidden_neuron.weight_gradients[i] += hidden_neuron.delta * input_activation
            # Accumulate bias gradient
            hidden_neuron.bias_gradients += hidden_neuron.delta

        #Update weights if batch is complete
        self.batch_number += 1
        if self.batch_number >= batch_size:
            self.updateWeights(batch_size, learning_rate)
            self.batch_number = 0

    def testNetwork(self, testing_data, correct_label_list):
        '''
        Tests a network on a dataset
        '''
        correct = 0
        i = 0
        for datapoint in range(testing_data):
            outputs = self.forwardPass(datapoint)
            predictions = getPredictionClass(outputs)
            if predictions == correct_label_list[i]:
                correct += 1
            1 += i
        accuracy = correct / len(testing_data)
        print(f"Test accuracy: {accuracy*100:.2f}%")
        return accuracy

    def trainAndEvaluate(self, test_data, training_data, testing_label_list, training_label_list, epochs, batch_size, learning_rate, return_full_history=False):
        """
        Trains the network while recording test accuracy each epoch,
        then plots accuracy vs epoch and prints the max accuracy achieved.
        """
        for epoch in range(epochs):
            #Shuffle the data each epoch for better results
            training_data, training_label_list = shuffleData(training_data, training_label_list)
            
            #Forward pass and back propogate for each datapoint
            datapoint_index = 0
            for datapoint in training_data:
                outputs = self.forwardPass(datapoint)
                correct_label = training_label_list[datapoint_index]
                self.backPropagate(outputs, correct_label, learning_rate, batch_size)
                datapoint_index += 1

            #Handles weight updates that have not been processed
            if self.batch_number > 0:
                self.updateWeights(self.batch_number, learning_rate)
                self.batch_number = 0
            
            #Calculate test accuracy for this epoch
            test_accuracies_list = []
            accuracy = self.testNetwork(test_data, testing_label_list)
            test_accuracies_list.append(accuracy)
            print(f"Epoch {epoch+1}: Test Accuracy = {accuracy:.4f}")

        #Print maximum accuracy achieved
        max_accuracy = max(test_accuracies_list)
        print(f"Maximum Test Accuracy Achieved: {max_accuracy:.4f}")

        #Handle what metric is needed
        if return_full_history:
            return test_accuracies_list
        else:
            return max_accuracy

#---Helper Functions---#

#Data Handling
def loadDataset(filename):
    """
    Load CSV.gz dataset into a list of rows
    """
    with gzip.open(filename, 'rt') as f:
        # skip the header row (labels) with skiprows=1
        data = np.loadtxt(f, delimiter=",", skiprows=1, dtype=int)
    return data.tolist()

def generateCorrectLabels(dataset, output_neuron_amount):
    '''
    Converts categorical labels to one-hot encoded lists
    '''
    correct_label_list = []
    for datapoint in dataset:
        label = datapoint[0]  # first column is the class label
        one_hot = [0] * output_neuron_amount
        one_hot[label] = 1
        correct_label_list.append(one_hot)

    return correct_label_list

def shuffleData(data, labels):
    """
    Shuffle data and labels
    """
    combined = list(zip(data, labels))
    random.shuffle(combined)
    shuffled_data, shuffled_labels = zip(*combined)
    return list(shuffled_data), list(shuffled_labels)

#Network Generation
def generateRandomWeight(lower_bound, upper_bound):
    '''
    Generates a random weight given a range
    '''
    return random.uniform(lower_bound, upper_bound)

def generateInputLayer(input_neurons):
    '''
    Generates the input layer with a variable amount of input neurons
    '''
    
    input_layer_neurons =[]
    for i in range(input_neurons):
        input_layer_neurons.append(Neuron())

    return NetworkLayer(input_layer_neurons, size=input_neurons)

def generateHiddenLayer(hidden_neuron_amount, input_layer_neuron_amount):
    '''
    Generates the hidden layer with a variable amount of hidden neurons
    '''
    hidden_layer_neurons =[]
    for i in range(hidden_neuron_amount):
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
    for i in range(output_neuron_amount):
        weight_list = []
        for i in range(hidden_layer_neuron_amount):
            weight_list.append(generateRandomWeight(-0.01, 0.01))
        output_layer_neurons.append(Neuron(weight_list, 0.0))
    return NetworkLayer(output_layer_neurons)

#Utilities
def getPredictionClass(outputs_activations_list):
    '''
    Interprets the prediction of a forward pass given an activation list, 
    returns a list with 1 at the prediction neuron index
    '''
    prediction_label_list = [0] * len(outputs_activations_list)
    max_index = outputs_activations_list.index(max(outputs_activations_list))
    prediction_label_list[max_index] = 1

    return prediction_label_list

#Visulisation/Testing
def plotMultiple(test_accuracy_list, filename, graph_title, metric_name, metrics_list):
    plt.figure(figsize=(12, 8))
    
    if metrics_list:
        for i, row in enumerate(test_accuracy_list):
            if i < len(metrics_list):
                plt.plot(range(1, len(row)+1), row, label=f"{metric_name}: {metrics_list[i]}")
            else:
                plt.plot(range(1, len(row)+1), row, label=f"Test {i+1}")
    else:
        for i, row in enumerate(test_accuracy_list):
            plt.plot(range(1, len(row)+1), row, label=f"Test {i+1}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"{graph_title}")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

def requiredTestGraphs(NInput, NHidden, NOutput, training_data, testing_data, testing_label_list, training_label_list):
    '''
    Runs the assignment required tests and saves the graphs to ./
    '''

    batch_size = 20
    epochs = 30
    learning_rate = 3

    # Testing learning rates
    print("Training and testing with different learning rates")
    test_learning_rate = 0.001
    print(f"Learning rate = {test_learning_rate}")
    learning_test_accuracy_list = []
    for i in range(6):
        network = NeuralNetwork([generateInputLayer(NInput), 
                            generateHiddenLayer(NHidden, NInput), 
                            generateOutputLayer(NOutput, NHidden)])

        max_accuracy = network.trainAndEvaluate(testing_data, training_data, testing_label_list, training_label_list, epochs, batch_size, test_learning_rate)
        learning_test_accuracy_list.append(max_accuracy)

        test_learning_rate *= 10
        print(f"Learning rate = {test_learning_rate}")
    print("Learning rate tests complete")
    learning_rates_used = [0.001, 0.01, 0.1, 1.0, 10, 100]
    plotMultiple(learning_test_accuracy_list, "Learning_Rates_Comparison.png", 
             "Model Accuracy vs Learning Rate", "Learning Rate", metrics_list=learning_rates_used)

    # Testing mini-batch sizes
    print("Training and testing with different batch sizes")
    test_batch_sizes = [1, 5, 20, 100, 300]
    batch_test_accuracy_list = []
    for i in range(5):
        test_batch_size = test_batch_sizes[i]
        print(f"Batch size = {test_batch_size}")
        network = NeuralNetwork([generateInputLayer(NInput), 
                            generateHiddenLayer(NHidden, NInput), 
                            generateOutputLayer(NOutput, NHidden)])
        
        max_accuracy = network.trainAndEvaluate(testing_data, training_data, testing_label_list, training_label_list, epochs, test_batch_size, learning_rate)
        batch_test_accuracy_list.append(max_accuracy)
        
    print("Batch size tests complete")
    plotMultiple(batch_test_accuracy_list, "Batch_Size_Comparison.png",
                 "Model Accuracy vs Batch Size", "Batch Size", metrics_list=test_batch_sizes)

def main():
    #Handle command line arguments
    if len(sys.argv) != 6:
        print("Usage: python nn.py NInput NHidden NOutput train.csv.gz test.csv.gz")
        sys.exit(1)
    NInput = int(sys.argv[1])
    NHidden = int(sys.argv[2])
    NOutput = int(sys.argv[3])
    train_file = sys.argv[4]
    test_file = sys.argv[5]

    # Load datasets
    training_data = loadDataset(train_file)
    testing_data = loadDataset(test_file)
    print(f"Loaded {len(training_data)} training examples")
    print(f"Loaded {len(testing_data)} test examples")
    training_label_list = generateCorrectLabels(training_data, NOutput)
    testing_label_list = generateCorrectLabels(testing_data,  NOutput)

    #Build network
    print(f"Networks build with {NInput} input, {NHidden} hidden, {NOutput} output neurons")
    network = NeuralNetwork([generateInputLayer(NInput), 
                             generateHiddenLayer(NHidden, NInput), 
                             generateOutputLayer(NOutput, NHidden)])
    
    #Training settings
    batch_size = 20
    epochs = 15
    learning_rate = 3

    #Train network
    print(f"Training and testing with default settings: Epochs = {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate")
    test_accuracies_list = network.trainAndEvaluate(testing_data, training_data, testing_label_list, training_label_list, epochs, batch_size, learning_rate, return_full_history=True)

    #Plot accuracy vs epoch
    plt.figure()
    plt.plot(range(1, epochs+1), test_accuracies_list, marker='o')
    plt.title("Test Accuracy vs Epoch - Network [784, 30, 10]")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.savefig("experiment1_accuracy_vs_epoch.png")

    #Additional Tests
    prompt = input("Run full graph generation Y/N? (N) ").strip().upper()
    if prompt == "Y":
        print("Running additonal tests")
        requiredTestGraphs(NInput, NHidden, NOutput, training_data, testing_data, testing_label_list, training_label_list)

if __name__ == "__main__":
    main()