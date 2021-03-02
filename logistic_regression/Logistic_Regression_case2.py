import numpy as np
import matplotlib.pyplot as plt
# let w and or_feature[pos] dot => n = w0x0 + w1x1 + w2x2 
def Dot(w,or_feature):
    return np.dot(w,or_feature)
# using Dot result, let it become a value that 0 ~ 1
def Sigmoid(w,or_feature):
    return ( 1 / (1 + np.exp( (-1) * Dot(w, or_feature))) )

def Gradient_Descent(w, or_feature, or_label, lr):
    return w + lr * (or_label - Sigmoid(w,or_feature)) * or_feature

def Cross_Entropy(or_label, or_feature, w):
    a = Sigmoid(w, or_feature)
    return -(or_label * np.log(a) + (1 - or_label) * np.log(1 - a) ) 

def Draw(w, or_feature, Epoch):
   
    x = np.linspace(-1,1) 
    # y is dependent variable
    y =  (w[1] * x + w[0]) *(-1) /(w[2]) 
    # show decision line
    plt.plot(x,y) 
    # show or data
    plt.scatter(or_feature[1:,1], or_feature[1:,2], label= "Training Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(or_feature[0,1], or_feature[0,2], label= "Training Data = 0", color= "green", marker= "+", s=30)
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Decision Line and OR Data')  
    plt.legend() 
    plt.show() 

def main():
    or_feature = np.array([[1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1]])
    or_label = np.array([0,1,1,1])
    w = [0.0, 0.0, 0.0]    
    lr = 0.1
    Epoch = 50000
    print("Max Epoch is", Epoch-1)
    # train weight
    for i in range(Epoch):
        loss = 0.0
        for pos in range(4):
            loss += Cross_Entropy(or_label[pos], or_feature[pos], w)
            w = Gradient_Descent(w, or_feature[pos], or_label[pos], lr)

        if ( loss / 4 )  <= 0.001 :
            print("Epoch = ", i )
            print("loss = ",loss/4,"less than 0.001")
            break

    # print  w0x0 + w1x1 + w2x2 
    print("%.3f + %.3f * x1 + %.3f * x2 = 0" %(w[0],w[1],w[2])) 
    Draw(w, or_feature, Epoch)

main()
