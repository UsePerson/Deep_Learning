import numpy as np
import matplotlib.pyplot as plt
# let w and and_feature[pos] dot => n = w0x0 + w1x1 + w2x2 
def Dot(w,and_feature):    
    return np.dot(w,and_feature)
# using Dot result, let it become a value that 0 ~ 1
def Sigmoid(w,and_feature): 
    return ( 1 / (1 + np.exp( (-1) * Dot(w,and_feature))) )

def Gradient_Descent(w, and_feature, and_label, lr):
    return w + lr * (and_label - Sigmoid(w,and_feature)) * and_feature

def Cross_Entropy(and_label, and_feature, w):
    a = Sigmoid(w,and_feature)
    return -(and_label * np.log(a) + (1 - and_label) * np.log(1 - a) ) 

def Draw(w, and_feature):
   
    x = np.linspace(0,1) 
    # y is dependent variable
    y =  (w[1] * x + w[0]) *(-1) /(w[2]) 
    # show decision line
    plt.plot(x,y) 
    # show AND data
    plt.scatter(and_feature[3,1], and_feature[3,2], label= "Training Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(and_feature[0:3,1], and_feature[0:3,2], label= "Training Data = 0", color= "green", marker= "+", s=30)
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Decision Line and AND Data')  
    plt.legend() 
    plt.show() 
    
def main():
    and_feature = np.array([[1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1]])
    and_label = np.array([0,0,0,1])
    w = [0.0, 0.0, 0.0]    
    lr = 0.1
    Epoch = 50000
    print("Max Epoch is", Epoch-1)
    # train weight
    for i in range(Epoch):
        loss = 0.0
        for pos in range(4): 
            loss += Cross_Entropy(and_label[pos], and_feature[pos], w)
            w = Gradient_Descent(w, and_feature[pos], and_label[pos], lr)

        if ( loss / 4 )  <= 0.001 :
            print("Epoch = ", i )
            print("loss = ",loss/4,"less than 0.001")
            break
    
    # print  w0x0 + w1x1 + w2x2 
    print("%.3f + %.3f * x1 + %.3f * x2 = 0" %(w[0],w[1],w[2])) 
    Draw(w, and_feature)

main()
