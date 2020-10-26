import numpy as np
import matplotlib.pyplot as plt

def sign(output):
    if(output > 0):
        return 1
    elif(output == 0):
        return 0
    else:
        return -1

def process_train():
    global train
    global train_label
    global train_feature
    global w
    # read train.txt
    train = np.loadtxt('./data/train.txt',delimiter=',')
    # train_label get label 1 or -1
    train_label = train[:,-1]
    # train_feature get features and insert a column of 1 to first column
    train_feature = train[:,0:2]
    train_feature = np.insert(train_feature,0,np.ones(len(train_feature)),axis=1)
    # weight w
    w = [0,0,0]

def pla():
    global w
    cnt = 0
    # revise w
    # if all correct, end loop
    while(cnt != len(train_feature)):
        cnt = 0
        
        for i in range(len(train_feature)):
            if( sign( int(np.inner(w , train_feature[i])) )!= train_label[i]):
                w += train_feature[i] * train_label[i]
            else:
                cnt+=1

def process_test():
    global test
    global test_feature
    global test_label
    test_label=[]
    # read test.txt 
    test = np.loadtxt('./data/test.txt',delimiter=',')
    test_feature = test
    # test_feature get (x1, x2)
    test_feature = np.insert(test_feature,0,np.ones(len(test_feature)),axis=1)
    # using inner product w and test_feature to get label
    
    for i in range(len(test_feature)):
        test_label=np.append(test_label,sign( int(np.inner(w , test_feature[i])) ))

def printOut():
    global w
    # print weight w
    print("{} + {} * x1 + {} * x2 = 0".format(int(w[0]),int(w[1]),int(w[2])))
    print(np.hstack((test_feature[:,1:3],test_label[:,np.newaxis])))

def draw():
    global w
    # x is indenpent variable
    x = np.linspace(-25, 25, 20) 
    # y is dependent variable
    y =  (w[1] * x + w[0]) *(-1) /(w[2]) 
    # draw line
    plt.plot(x,y) 
    # draw train data position 
    train_positive = train[train[:,2].astype(int)==1,:]
    train_negative = train[train[:,2].astype(int)==-1,:]
    plt.scatter(train_positive[:,0], train_positive[:,1], label= "Training Data = 1", color= "green", marker= "*", s=30) 
    plt.scatter(train_negative[:,0], train_negative[:,1], label= "Training Data = -1", color= "green", marker= "+", s=30) 
    # draw test data position 
    test = np.hstack((test_feature[:,1:3],test_label[:,np.newaxis])) 
    test_positive = test[test[:,2].astype(int)==1,:]
    test_negative = test[test[:,2].astype(int)==-1,:]
    plt.scatter(test_positive[:,0], test_positive[:,1], label= "Testing Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(test_negative[:,0], test_negative[:,1], label= "Testing Data = -1", color= "red", marker= "+", s=30) 
    
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Training and Testing Data')  
    plt.legend() 
    plt.show() 


process_train()
pla()
process_test()
printOut()
draw()



