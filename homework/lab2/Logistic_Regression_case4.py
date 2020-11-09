import numpy as np
import matplotlib.pyplot as plt
# let w and train_feature[pos] dot => n = w0x0 + w1x1 + w2x2 
def Dot(w,train_feature):
    return np.dot(w,train_feature)
# using Dot result, let it become a value that 0 ~ 1
def Sigmoid(w,train_feature):
    dot = Dot(w,train_feature)
    if -dot > np.log(np.finfo(type(dot)).max): 
        return 0.0    
    return 1.0/(1.0 + np.exp(-dot))
 
def Gradient_Descent(w, train_feature, train_label, lr):
    return w + lr * ( train_label - Sigmoid(w,train_feature) ) * train_feature
# i find a problem that ln divide by zero
# def Cross_Entropy(train_label, train_feature, w):
#     a = Sigmoid(w,train_feature)
#     return -( train_label * np.log(a) + (1 - train_label) * np.log(1 - a)) 

def Draw(w, train_feature, test_feature, test_label):
    x = np.linspace(0,180) 
    # y is dependent variable
    y =  -( (w[1] * x + w[0]) /(w[2]) ) 
    # show decision line
    plt.plot(x,y) 
    # show train data    
    plt.scatter(train_feature[0:6,1], train_feature[0:6,2], label= "Training Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(train_feature[6:,1], train_feature[6:,2], label= "Training Data = 0", color= "red", marker= "+", s=30)
    # show test data 
    test = np.hstack((test_feature[:,1:],test_label[:,np.newaxis])) 
    print(test[:,:])
    test_positive = test[test[:,2].astype(int)==1,:]
    test_negative = test[test[:,2].astype(int)==0,:]
    plt.scatter(test_positive[:,0], test_positive[:,1], label= "Testing Data = 1", color= "green", marker= "*", s=30) 
    plt.scatter(test_negative[:,0], test_negative[:,1], label= "Testing Data = 0", color= "green", marker= "+", s=30) 
    
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Training and Testing Data')  
    plt.legend() 
    plt.show() 

def main():
    train_feature = np.array([[1,170,80],
                              [1,165,55],
                              [1,150,45],
                              [1,180,70],
                              [1,175,65],
                              [1,160,60],
                              [1,90,15],
                              [1,130,30],
                              [1,120,40],
                              [1,110,35]])
    train_label = np.array([1,1,1,1,1,1,0,0,0,0])
    test_feature = np.array([[1,170,60],
                              [1,85,15],
                              [1,145,45]])
    test_label = np.array([])
    w = np.array([0.0, 0.0, 0.0])    
    lr = 0.01 
    Epoch =1000001
    print("Max Epoch is", Epoch-1)
    # train weight
    for i in range(Epoch):
        for pos in range(10):
            w = Gradient_Descent(w, train_feature[pos], train_label[pos], lr)
    # print  w0x0 + w1x1 + w2x2 
    print("%.3f + %.3f * x1 + %.3f * x2 = 0" %(w[0],w[1],w[2])) 

    # calculate the test data
    for pos in range(3):    
        if( Sigmoid(w,test_feature[pos]) >= 0.5):
            test_label = np.append(test_label,1)
        else:
            test_label = np.append(test_label,0)

    Draw(w, train_feature, test_feature, test_label)

main()