import numpy as np


class ErrorFunction:
    
    def __init__(self, types='cross_entropy'):
        
        self.func = self.cross_entropy
        self.dfunc = self.dcross_entropy
        
        if types == 'MSE':
            self.func = self.MSE
            self.dfunc = self.dMSE
            
            
    def cross_entropy(self, ans, output):
        
        return -( np.multiply(ans, np.log(output)) + np.multiply((1 - ans), np.log(1 - output) ))
    
    def dcross_entropy(self, ans, output):
        
        return (output - ans) / np.multiply(output, (1 - output))
    
    def MSE(self, ans, output):
        
        return ((ans - output)**2)/2
    
    def dMSE(self, ans, output):
        
        return -(ans - output)
    
if __name__ == '__main__':
    
    myfunc = ErrorFunction('cross_entropy')






