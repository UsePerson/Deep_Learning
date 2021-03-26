import numpy as np
import matplotlib.pyplot as plt 
from activityFunction import ActivationFunction as AF
from errorFunction import ErrorFunction as EF

"""
store each layer state

contains weight, bias, output, delta, ...

and feedforwd , update, calculate delta
"""

class Layer_state:
    
    def __init__(self, input_size, neuron_size, af, ef, lr):
        
        self.__neuron_size = neuron_size
        self.__input_size = input_size
        self.__weight = np.random.randn(neuron_size, input_size) 
        self.__bias = np.random.randn(neuron_size,1) 
        self.__af = AF(types=af)
        self.__ef = EF(types=ef)
        self.__output = np.array([])
        self.__delta = np.array([])
        self.__lr = lr
        
    @property
    def delta(self):
        
        return self.__delta
    
    @property
    def weight(self):
        
        return self.__weight
    
    @property
    def bias(self):
        
        return self.__bias
    
    @property
    def output(self):
        
        return self.__output
    
    @output.setter
    def output(self, output):
    
        self.__output = output
    
    @property
    def inputs(self):
        
        return self.__input_size
    
    @property
    def neurons(self):
        
        return self.__neuron_size
    
    def feed_forward(self, inputs):
        
        n = self.weight.dot(inputs) + self.bias
        self.output = self.__af.func(n) 
        
        return self.__output
    
    def calculate_last_delta(self, ans):
        
        self.__delta = self.__af.dfunc(self.__output) * self.__ef.dfunc(ans, self.__output) 
        
    def calculate_delta(self, weight, delta):
        
        self.__delta = weight.T.dot( delta ) * self.__af.dfunc(self.__output)
        
    def update(self, inputs):

        self.__weight = self.__weight - self.__lr * self.__delta.dot(inputs.T)
        self.__bias = self.__bias - self.__lr * self.__delta
    
    
class nn:
    
    def __init__(self, ef='cross_entropy', lr=0.5 ):
        
        self.ef = EF(types=ef)
        self.lr = lr
        self.layer_state = np.array([])
    
    def add_layer(self, input_size, neuron_size, af='Sigmoid' ):
        
        state = Layer_state(input_size, neuron_size, af, self.ef, self.lr)
        self.layer_state = np.append(self.layer_state, state) 
        
    def feedforward(self, inputs):
        
        data = np.array(inputs[0 : self.layer_state[0].inputs ])
        data = data.reshape(-1, data.ndim)
        
        for Layer in self.layer_state:
                
            data = Layer.feed_forward(data)
            
        return data
    
    
    def calculate_delta(self, ans):
        
        """ last delta = derivative error function * derivative activity function """ 
        self.layer_state[-1].calculate_last_delta(ans)
        
        i = np.size(self.layer_state) - 2 # sub 2, because have already calcutaled the last delta
        
        while i >= 0:
            
            """ delta(i) = weight * delta(i+1) * derivative activity function """ 
            self.layer_state[i].calculate_delta(self.layer_state[i+1].weight, self.layer_state[i+1].delta)
            i -= 1
            
    def update_weights(self, inputs):
        """
         update first weight and bias
         -----------------------------
         Formula :
             weight = weight - learning * delta * input
             bias = bias - learning rate * delta
        """
        self.layer_state[0].update(inputs)
        
        """
        update other weight and bias
        -----------------------------
        Formula :
            weight(i) = weight - learning * delta(i) * input(i - 1)
            bias(i) = bias - learning rate * delta(i) 
        """
        i = 1    
        while i < np.size(self.layer_state):
            
            self.layer_state[i].update(self.layer_state[i-1].output)
            i += 1
    
    def backpropagation(self, dataset):
        
        """
        get the train feature and label
        """
        inputs = np.array( dataset[0:self.layer_state[0].inputs] )
        output = np.array( dataset[self.layer_state[0].inputs:] )
        
        inputs = inputs.reshape(-1, inputs.ndim)
        output = output.reshape(-1, output.ndim)
        
        self.calculate_delta(output)
        self.update_weights(inputs)
    
    def accuracy(self, output, ans):
        
        """
        get the train label
        """
        ans = np.array( ans[self.layer_state[0].inputs:] )
        ans = ans.reshape(-1, ans.ndim)
        
        
        if np.array_equal(np.where(output == output.max()), np.where(ans == ans.max())) :
            
            return 1
        
        return 0

    def loss(self, output, ans):
        
        ans = np.array( ans[self.layer_state[0].inputs:] )
        ans = ans.reshape(-1, ans.ndim)
        
        return self.ef.func(ans, output)
    
    def draw_accuracy(self, x, train_y, valid_y, string):
        
        train_str = "Train " + string
        valid_str = "Validation " + string
        title_str = "Each epoch\'s " + string
        
        plt.plot(x, train_y, label= train_str)
        plt.plot(x, valid_y, label= valid_str)
        plt.xlabel('Epoch')
        plt.ylabel(string)
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    def SGD(self, inputs, validation_rate, epoch):
        
        """
        get the train data size 
        
        get the validation data size
        """
        train_size = int(np.size(inputs, 0) * (1 - validation_rate))
        validation_size = int(np.size(inputs, 0)) - train_size
           
        """
        get the train data  
        
        get the validation data
        """    
        training_data =  np.array(inputs[ 0 : train_size ])
        validation_data = np.array(inputs[ train_size :   ])
        
        train_accuracy = np.array([])
        validation_accuracy = np.array([])
        
        train_loss = np.array([])
        validation_loss = np.array([])
        
        mini_loss = 100
        mini_loss_epoch = 0
        
        for i in range(epoch):
            
            train_accurate_counter = 0
            validation_accurate_counter = 0
            
            train_loss_p = 0
            validation_loss_p = 0
            
            """
            train data using (1 - validation_rate ) percentage of input data
            
            using feedforward, calculate the loss and accuray
            
            then backpropagation, update weights and bias
            """
            for data in training_data:
                
                output = self.feedforward(data)
                train_accurate_counter += self.accuracy(output, data)
                train_loss_p = train_loss_p + np.sum( self.loss(output, data) )
                self.backpropagation(data)
            
            train_loss = np.append(train_loss, (train_loss_p / train_size) )
            train_accuracy = np.append(train_accuracy, (train_accurate_counter / train_size) )

            """
            validation data using (validation_rate) percentage of input data 
            
            calculate the validation data accuracy
            
            not using backpropagation, only feedforward
            """
            for data in validation_data :
                
                output = self.feedforward(data)
                validation_accurate_counter += self.accuracy(output, data)
                validation_loss_p = validation_loss_p + np.sum( self.loss(output, data) )
                
            validation_loss = np.append(validation_loss, (validation_loss_p / validation_size) )    
            validation_accuracy = np.append(validation_accuracy, (validation_accurate_counter / validation_size) )    
            
            if (mini_loss > (train_loss_p / train_size)) :
                
                mini_loss = train_loss_p / train_size
                mini_loss_epoch = i
            
            if ( (train_loss_p / train_size) < 0.05) or ( (validation_loss_p / validation_size) < 0.05 ) :
                print("i :", i)
                print("train loss : ", train_loss_p / train_size )
                print()
                print("validation loss : ", validation_loss_p / validation_size )
                print("--------------------------")
                break
                
        self.draw_accuracy(range(epoch), train_accuracy, validation_accuracy, "Accuracy")
        self.draw_accuracy(range(epoch), train_loss, validation_loss, "Loss")
        print("mini Epoch : ", mini_loss_epoch)
        print("mini loss : ", mini_loss)
        
    def test(self, test):
        
        test_size = np.size(test, 0)
        test_accurate_counter = 0
        
        for data in test :
            
            output = self.feedforward(data)
            test_accurate_counter += self.accuracy(output, data)
            
        print("test accuracy : ", test_accurate_counter / test_size )
        
        
