import numpy as np

class ActivationFunction:
    
    def __init__(self, types='Sigmoid'):
        
        self.func = self.Sigmoid
        self.dfunc = self.dSigmoid               
        
    def Sigmoid(self, n):

        return 1.0 / (1.0 + np.exp(-n))
    
    def dSigmoid(self, output):
        
        return output * (1 - output)
    
if __name__ == '__main__':
    
    myfunc = ActivationFunction('Sigmoid')
    
    



