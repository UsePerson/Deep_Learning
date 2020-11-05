import numpy as np
import matplotlib.pyplot as plt
# let w and xor_feature[pos] dot => n = w0x0 + w1x1 + w2x2 
def Dot(w, xor_feature):
    return np.dot(w, xor_feature)
# using Dot result, let it become a value that 0 ~ 1
def Sigmoid(w, xor_feature):
    return ( 1 / (1 + np.exp( (-1) * Dot(w, xor_feature))) )
 
def Gradient_Descent(w, xor_feature, xor_label, lr):
    return w + lr * (xor_label - Sigmoid(w,xor_feature)) * xor_feature

def Draw_and_Print(w, xor_feature, Epoch):
    # print  w0x0 + w1x1 + w2x2  
    print("%.5f + %.5f * x1 + %.5f * x2 = 0" %(w[0],w[1],w[2])) 
    # print Epoch
    print("Epoch : ", Epoch)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    x1 = np.linspace(-1,3) 
    # y is dependent variable
    y1 =  (w[1] * x1 + w[0]) *(-1) /(w[2])
    # show decision line 
    plt.plot(x1,y1) 
    plt.scatter(xor_feature[2:,1], xor_feature[2:,2], label= "Training Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(xor_feature[0:2,1], xor_feature[0:2,2], label= "Training Data = 0", color= "green", marker= "+", s=30)
    # show XOR data
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Decision Line and XOR Data')  
    plt.legend() 
    plt.show() 

def main():    
    xor_feature = np.array([[1,0,0],
                  [1,1,1],
                  [1,1,0],
                  [1,0,1]])
    xor_label = np.array([0,0,1,1])
    w = np.array([0.0, 0.0, 0.0])    
    lr = 0.5
    Epoch = 50
    
    for i in range(Epoch):
        for pos in range(4):
            w = Gradient_Descent(w, xor_feature[pos], xor_label[pos], lr)
            
    Draw_and_Print(w, xor_feature, Epoch)

main()

