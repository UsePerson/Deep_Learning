import matplotlib.pyplot as plt 
import numpy as np
w0,w1,w2=0,0,0
testing_label =[]
def sign(f):
    if(f > 0):
        return 1
    elif(f == 0):
        return 0
    else:
        return -1
with open("train.txt","r") as train_file:   # read train.txt, first column is x1 and second cloume is x2
    train_x1,train_x2,label = [],[],[]       
    train_positive_x1,train_positive_x2 = [],[]    
    train_negative_x1,train_negative_x2 = [],[]
    test_positive_x1,test_positive_x2 =[],[]
    test_negative_x1,test_negative_x2 =[],[]
    for line in train_file:
        try:
            split = line.split(',')
            train_x1.append(int(split[0]))  # train_x1 get all x1
            train_x2.append(int(split[1]))  # train_x2 get all x2
            label.append(int(split[2]))
            if(int(split[2]) == 1 ):
                train_positive_x1.append(int(split[0]))  # train_positive_x1 get x1 that label is 1
                train_positive_x2.append(int(split[1]))  # train_positive_x2 get x2 that label is 1
            else:
                train_negative_x1.append(int(split[0]))  # train_negative_x1 get x1 that label is -1
                train_negative_x2.append(int(split[1]))  # train_negative_x2 get x2 that label is -1
        except:
            pass
    cnt =0
    # if all label match, will break
    while cnt != len(train_x1):  
        cnt = 0
        for i in range(len(train_x1)):
            try:
                if(sign( w0 + w1 * train_x1[i] + w2 * train_x2[i] ) != label[i]):   # if w0*x1 + w1*x1 + w2*x2 not equal label, then vector w += label * vector x
                    w0 += label[i] * 1
                    w1 += label[i] * train_x1[i]
                    w2 += label[i] * train_x2[i]
                else :
                    cnt+=1
            except:
                pass
    print("{} + {}*x1 + {}*x2 = 0".format(w0,w1,w2)) # print w0*x1 + w1*x1 + w2*x2 = 0 
    with open("test.txt","r") as test_file: # read test.txt
        # calculate the test
        for line in test_file:  
            try:
                split = line.split(',')
                # print test x1, test x2 and test label
                print("\n{}, {}, {}".format(int(split[0]),int(split[1]), sign(w0 + w1 * int(split[0]) + w2 * int(split[1])))) 
                if(sign(w0 + w1 * int(split[0]) + w2 * int(split[1])) == 1 ): 
                    test_positive_x1.append(int(split[0]))  # if test label is equal 1, then test_positive_x1 get x1 
                    test_positive_x2.append(int(split[1]))  # if test label is equal 1, then test_positive_x2 get x2 
                else:
                    test_negative_x1.append(int(split[0]))  # if test label is equal -1, then test_positive_x1 get x1 
                    test_negative_x2.append(int(split[1]))  # if test label is equal -1, then test_positive_x2 get x2 
            except:
                pass
    # Draw Train and Test Data
    # x is indenpent variable
    x = np.linspace(-25, 25, 20) 
    # y is dependent variable
    y =  (w1 * x + w0) *(-1) /(w2) 
    # draw line
    plt.plot(x,y) 
    plt.scatter(train_positive_x1, train_positive_x2, label= "Training Data = 1", color= "green", marker= "*", s=30) 
    plt.scatter(train_negative_x1, train_negative_x2, label= "Training Data = -1", color= "green", marker= "+", s=30) 
    plt.scatter(test_positive_x1, test_positive_x2, label= "Testing Data = 1", color= "red", marker= "*", s=30) 
    plt.scatter(test_negative_x1, test_negative_x2, label= "Testing Data = -1", color= "red", marker= "+", s=30) 
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.title('Training and Testing Data')  
    plt.legend() 
    plt.show() 