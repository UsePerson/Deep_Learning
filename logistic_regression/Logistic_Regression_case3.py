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

def Cross_Entropy(xor_label, xor_feature, w):
    a = Sigmoid(w,xor_feature)
    return -(xor_label * np.log(a) + (1 - xor_label) * np.log(1 - a) ) 

def Draw(w, xor_feature):
    
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
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
    Epoch = 50000
    print("Max Epoch is", Epoch-1)
    min_loss = 1.0
    is_maxE = True
    for i in range(Epoch):
        loss = 0.0
        for pos in range(4):
            loss += Cross_Entropy(xor_label[pos], xor_feature[pos], w)
            w = Gradient_Descent(w, xor_feature[pos], xor_label[pos], lr)
       
        min_loss = min((loss/4),min_loss)
        if ( loss / 4 )  <= 0.001 :
            print("Epoch = ", i )
            print("loss = ",loss/4,"less than 0.001")
            is_maxE = False
            break 

    if is_maxE == True :
        print("Epoch = ", i )
        print("no less than 0.001")
    Draw(w, xor_feature)

main()

