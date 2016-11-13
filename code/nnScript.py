import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import os
import time
import pickle


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig= 1/(1+ np.exp(-1*z))
    return sig
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    mat = loadmat('mnist_all.mat')
    print(mat.keys())
    
    #Stack all training matrices into one 60000  784 matrix. Do the same for test matrices.
    stackTrainingData = mat.get('train0')
    stackTestData = mat.get('test0')
    for i in range(1,10):     
        stackTrainingData = np.concatenate((mat.get('train'+str(i)),stackTrainingData),axis = 0)
        stackTestData = np.concatenate((mat.get('test'+str(i)),stackTestData),axis = 0)
        
    #np.savetxt('test.txt', stackTestData)
    
    #Create a 60000 length vector with true labels (digits) for each training example. Same for test data.
    
    training_label = np.zeros((len(stackTrainingData),10))
    testing_label = np.zeros((len(stackTestData),10))
    
    temp1=0;
    for i in range(0,10):
        rowsStackTrainingData = len(mat.get('train' + str(i)))
        for j in range(rowsStackTrainingData):
            training_label[j+temp1,i] = 1
        temp1 = temp1 + rowsStackTrainingData;
        
    temp2=0;
    for i in range(0,10):
        rowsStackTestData = len(mat.get('test' + str(i)))
        for j in range(rowsStackTestData):
            testing_label[j+temp2,i] = 1
        temp2 = temp2 + rowsStackTestData; 
        
    #Normalize the training matrix and test matrix so that the values are between 0 and 1 and randomly split the normalized training matrix into two matrices
    normalizedTrainingData = np.true_divide(stackTrainingData, 255.0)
    normalizedTestData = np.true_divide(stackTestData, 255.0)
    randomNumRange = range(len(stackTrainingData))
    randomNums = np.random.permutation(randomNumRange)
    randomTrainingData = normalizedTrainingData[randomNums[0:50000],:]
    randomValidationData = normalizedTrainingData[randomNums[50000:],:]
    randomTrainingLabel = training_label[randomNums[0:50000],:]
    randomValidationLabel = training_label[randomNums[50000:],:]
    
    #Your code here
    train_data = randomTrainingData
    train_label = randomTrainingLabel
    validation_data = randomValidationData
    validation_label = randomValidationLabel
    test_data = normalizedTestData
    test_label = testing_label
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Create a column for bias node and append it to our training data. Evaluate Sigmoid for hidden and output nodes
    biasInputNodeData = np.ones((len(training_data),1))
    final_training_data = np.concatenate((training_data, biasInputNodeData), axis=1)
    input_hidden_sigmoid = np.dot(final_training_data, w1.T)
    output_hidden_sigmoid = sigmoid(input_hidden_sigmoid)
    biasHiddenNodeData = np.ones((len(output_hidden_sigmoid),1))
    final_hidden_data = np.concatenate((output_hidden_sigmoid, biasHiddenNodeData), axis=1)
    input_final_sigmoid = np.dot(final_hidden_data, w2.T)
    output_final_sigmoid = sigmoid(input_final_sigmoid)
    
    #Calculating the regularised error value and gradience with respect to the second weight vector w2
    diffMatrix = training_label - output_final_sigmoid #Calculating (yl-ol)
    squareDiffMatrix = np.square(diffMatrix) #Calculating square of (yl-ol)
    totalErrorValue = ((np.sum(squareDiffMatrix))/(2*(len(training_data))))
    w2Squared = np.square(w2)
    w2SquaredSum = np.sum(w2Squared)
    w1Squared = np.square(w1)
    w1SquaredSum = np.sum(w1Squared)
    effectiveLambda = ((lambdaval)/(2*(len(training_data))))
    regularisedErrorValue = ((w1SquaredSum + w2SquaredSum)*(effectiveLambda))
    obj_val = (totalErrorValue + regularisedErrorValue) # This is the final regularised error value
    unitMatrix1 = np.ones((len(training_data),10))
    compSigmoid = (unitMatrix1 - output_final_sigmoid)
    tempDelta = np.multiply(diffMatrix,compSigmoid)
    delta = np.multiply(output_final_sigmoid,tempDelta)
    
    # grad_w2 calculation
    w2GradianceSum = ((-1)*(np.dot(delta.T,final_hidden_data))) #Gradience sum without regularisation
    regFactor1 = (lambdaval*w2) # regularisation factor
    grad_w2 = (((w2GradianceSum) + (regFactor1))/(len(training_data)))
    
    #grad_w1 calculation
    deltaFactor = np.dot(delta,w2)
    unitMatrix2 = np.ones((len(training_data),(n_hidden+1)))
    compZFactor = (final_hidden_data - unitMatrix2)
    finalZFactor = np.multiply(final_hidden_data,compZFactor)
    backPropogationDelta = np.multiply(finalZFactor,deltaFactor)
    w1GradianceSum = np.dot(backPropogationDelta.T,final_training_data)
    w1GradianceSum = np.delete(w1GradianceSum,(len(w1GradianceSum)-1),0)
    regFactor2 = (lambdaval*w1) # regularisation factor
    grad_w1 = (((w1GradianceSum) + (regFactor2))/(len(training_data)))
        
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    biasInputNodeData = np.ones((len(data),1))
    final_training_data = np.concatenate((data, biasInputNodeData), axis=1)
    input_hidden_sigmoid = np.dot(final_training_data, w1.T)
    output_hidden_sigmoid = sigmoid(input_hidden_sigmoid)
    biasHiddenNodeData = np.ones((len(output_hidden_sigmoid),1))
    final_hidden_data = np.concatenate((output_hidden_sigmoid, biasHiddenNodeData), axis=1)
    input_final_sigmoid = np.dot(final_hidden_data, w2.T)
    output_final_sigmoid = sigmoid(input_final_sigmoid)
    labels=np.zeros((len(data),10))
    for i in range(output_final_sigmoid.shape[0]):
        index=np.argmax(output_final_sigmoid[i],axis=0)
        labels[i,index]=1
        
    #labels = output_final_sigmoid
    return labels
    



"""**************Neural Network Script Starts here********************************"""
begin =time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.7;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

end = time.time()
total_time_taken = begin - end
print('\n Total time taken by the program:' + str(total_time_taken) + 'milliseconds')

pickleFileData = { "n_hidden":50, "w1":w1, "w2":w2, "lambdaval":0.7}
pickle.dump(pickleFileData,open("params.pickle","wb"))

