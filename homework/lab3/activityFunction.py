import numpy as np

class ActivationFunction:
    
    def __init__(self, types='Sigmoid'):
        
        self.func = self.Sigmoid
        self.dfunc = self.dSigmoid
        
        if types == 'Sigmoid':
            
            self.func = self.Sigmoid
            self.dfunc = self.dSigmoid
            
        elif types == 'Sign':
            
            pass
        
        
        
        
    def Sigmoid(self, n):
        for i in range(x.shape[0]):
            if x[i]<-709.0:
                x[i]=-100.0
            elif x[i]>100.0:
                x[i]=100.0
        return 1 / (1 + np.exp(-n))
    
    def dSigmoid(self, output):
        
        return np.multiply(output, (1 - output))
    
if __name__ == '__main__':
    
    myfunc = ActivationFunction('Sigmoid')
    
    



