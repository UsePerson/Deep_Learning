import numpy as np
from activityFunction import ActivationFunction as AF
from errorFunction import ErrorFunction as EF

class nn:
    
    def __init__(self, af='Sigmoid', ef='cross_entropy', lr=0.1 ):
        
        self.af = AF(types=af)
        self.ef = EF(types=ef)
        self.lr = lr
        self.layer_state = np.array([])
    
    def add_layer(self, input_size, neuron_size ):
        
        state = Layer_state(input_size, neuron_size, self.af)
        self.layer_state = np.append(self.layer_state, state)        
     
    def feedforward(self, inputs):
        
        data = np.asmatrix(inputs[0,0:self.layer_state[0].inputs])
        for i, Layer in np.ndenumerate(self.layer_state):
                
            data = Layer.feed_forward(data)
            
        return data
    
    
    def calculate_delta(self,output):
        
        self.layer_state[-1].delta = np.multiply(self.af.dfunc(self.layer_state[-1].output), self.ef.dfunc(output, self.layer_state[-1].output))
        i = np.size(self.layer_state) - 2 # sub 2, because have already calcutaled the last delta
        
        while i >= 0:
            
            self.layer_state[i].delta = np.multiply( np.dot(self.layer_state[i+1].weight.T, self.layer_state[i+1].delta), self.af.dfunc(self.layer_state[i].output) )
            i -= 1
            
    def update_weights(self, inputs):
        
        self.layer_state[0].weight = self.layer_state[0].weight - self.lr * np.dot(self.layer_state[0].delta, inputs)
        self.layer_state[0].bias = self.layer_state[0].bias - self.lr * self.layer_state[0].delta
        i = 1    
        while i < np.size(self.layer_state):
            
            self.layer_state[i].weight = self.layer_state[i].weight - self.lr * np.dot(self.layer_state[i].delta, self.layer_state[i-1].output.T)
            self.layer_state[i].bias = self.layer_state[i].bias - self.lr * self.layer_state[i].delta
            i += 1
    
    def backpropagation(self, dataset):
        
        output = np.asmatrix(dataset[0,self.layer_state[0].inputs:])
        inputs = np.asmatrix(dataset[0,0:self.layer_state[0].inputs])

        self.calculate_delta(output.T)
        
        self.update_weights(inputs)
            
    def train(self, inputs, validation_rate, epoch):
        
        for i in range(epoch):
            
            for j in range(np.size(inputs, 0)):
                
                self.feedforward(np.asmatrix(inputs[j]))
                self.backpropagation(np.asmatrix(inputs[j]))
                

class Layer_state:
    
    def __init__(self, input_size, neuron_size, af):
        
        self.__neuron_size = neuron_size
        self.__input_size = input_size
        self.__weight = np.random.rand(neuron_size, input_size)
        self.__bias = np.random.rand(neuron_size,1)
        self.__af = af
        self.__output = np.array([])
        self.__delta = np.array([])
        
    @property
    def delta(self):
        
        return self.__delta
    
    @delta.setter
    def delta(self, delta):
        
        self.__delta = delta
    
    @property
    def inputs(self):
        
        return self.__input_size
    
    @property
    def neurons(self):
        
        return self.__neuron_size
    
    @property
    def weight(self):
        
        return self.__weight
    
    @weight.setter
    def weight(self, weight):
        
        self.__weight = weight
    
    @property
    def bias(self):
        
        return self.__bias
    
    @bias.setter
    def bias(self, bias):
        
        self.__bias = bias
    
    @property
    def output(self):
        
        return self.__output
    
    @output.setter
    def output(self, output):
    
        self.__output = output
    
    def feed_forward(self, inputs):
        
        self.output = self.__af.func(np.dot(self.weight, inputs.T) + self.bias)        
        return self.__output.T