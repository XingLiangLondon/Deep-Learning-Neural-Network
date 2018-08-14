################################################################################################
##       Deep Learning Neural Network for Nonlinear Regression using Gradient Descent         ##
##          [Linear->Leaky Relu]*(L-1)->[Linear->Pure Linear]*1                               ##
##                         by: Dr. Xing Liang                                                 ##
##                              06/08/2018                                                    ##     
################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv


############################ DEFINING THE TRAINNING MODEL #######################################
### Raw data Min-Max Normalized 
def meanNormalization (Xraw):   
    m=Xraw.shape[1]
    n=Xraw.shape[0]
    u=1/m*Xraw.sum(axis=1)
    u=u.reshape((n,1))
    xmax=Xraw.max(axis=1)
    xmax=xmax.reshape((n,1))
    xmin=Xraw.min(axis=1)
    xmin=xmin.reshape((n,1))
    Xnorm=(Xraw-u)/(xmax-xmin)
    return Xnorm

### initialize_parameters_deep 
def initialize_parameters_deep(layer_dims): 
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):    
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))     
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))  
    return parameters

###  linear_forward 
def linear_forward(A, W, b):
    Z = np.dot(W,A)+b 
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

### linear output y=z 
def linearOutput(z):   
    y = z
    cache =(z)
    return y, cache

### Activation FUNCTION: leaky relu forward 
def leakyrelu(z):
    a = np.maximum(0.01*z,z)
    cache =(z) 
    # a =  z * (z > 0)   # for RELU function
    return a, cache

### linear_activation_forward 
# linear_cache: A(l-1) (previous layer feeding in to current layer), Wl(current layer), bl(current layer)
# activation_cache: Zl(current layer)

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "leakyrelu":       
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leakyrelu(Z)
    
    elif activation == "linearOutput":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = linearOutput(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)      
    return A, cache  # return A(l), cache: (A(l-1), w(l), b(l), Z(l))

### L_model_forward 
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  
   
    # Implement [LINEAR -> leaky RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],"leakyrelu")
        caches.append(cache)
    
    # Implement [LINEAR ->linearOutput] at the output layer. Add "cache" to the "caches" list. 
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linearOutput")
    caches.append(cache)   
    assert(AL.shape == (y_train.shape[0], X.shape[1]))        
    return AL, caches

### compute_cost_without_L2Regularisation 
"""
def compute_cost(AL, Y):
    m = Y.shape[1] 
    # Compute Squares Error Cost Function from AL and Y.
    cost = 1/(2*m)*np.sum(np.power((AL-Y),2))
    cost = np.squeeze(cost)           
    assert(cost.shape == ())  
    return cost
"""

### compute_cost_with_L2Regularisation
def compute_cost_regularisation(AL, Y, parameters, lambd):
    m = Y.shape[1] 
    
    # Compute Squares Error Cost Function from AL and Y.
    cross_entropy_cost = 1/(2*m)*np.sum(np.power((AL-Y),2))
    
    # compute L2 regularisation cost
    L = len(parameters) // 2 
    L2 = 0
    
    for l in range(L):
        L2 += np.sum(np.square(parameters["W" + str(l+1)]))        
    L2_regularisation_cost = L2*(1/m)*(lambd/2)   
    
    cost = np.squeeze(cross_entropy_cost + L2_regularisation_cost)      
    assert(cost.shape == ())
    return cost

### linear_backward 
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
     
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
        
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

### leakyrelu_backward 
def leakyrelu_backward(dA, cache):
    Z=cache
    dg=np.ones_like(Z)
    dg[Z < 0] =0.01
    dZ=dA*dg
    
    ###This is for RELU
    #dZ= (Z > 0) * 1*dA 
    return dZ

### linearOutput_backward 
def linearOutput_backward(dA, cache):
    # Because dg=1, codes below are disabled to speed up
    """
    Z=cache 
    dg=1
    dZ=dA*dg
    """
    dZ=dA
    return dZ

### linear_activation_backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "leakyrelu":     
        dZ = leakyrelu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
          
    elif activation == "linearOutput":      
        dZ = linearOutput_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)          
    return dA_prev, dW, db

### L_model_backward 
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # Initializing backpropagation
    dAL = AL-Y
 
    # Lth layer (linearOutput -> LINEAR) gradients. 
    #Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]                          # caches[L-1]: A(L-1), w(L), b(L), Z(L)
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,"linearOutput")
     
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (leakyRELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "leakyrelu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads

### update_parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    # Update rule for each parameter.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)] 
    return parameters

### L_layer_model 
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations,lambd, print_cost):

    grads = {}
    costs = []                            
    m = X.shape[1]                        
           
    # Parameters Initialization 
    parameters = initialize_parameters_deep(layers_dims)   
     
    # Loop (gradient descent) 
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> leakyRELU ->LINEAR -> leakyRELU ->....... LINEAR -> linearOutput. 
        AL, caches= L_model_forward(X, parameters)
        
        # Compute cost 
        #cost = compute_cost(AL, Y)
        cost = compute_cost_regularisation(AL, Y, parameters, lambd)
   
               
        # Backward propagation 
        grads=L_model_backward(AL, Y, caches)
                
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example 
        if print_cost and i % 100 == 0:
           print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
           costs.append(cost)
              
    # plot cost 
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # print parameters 
    for keys,values in parameters.items():
        print(keys)
        print(values)
    return parameters

##############################   DEFINING THE DEV/TEST MODEL   ###################################
### Linear Forward
def linear_forward_test(A, W, b):
    Z = np.dot(W,A)+b 
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z

### Linear Output
def linearOutput_test(z):   
    y = z  
    return y

### Leakyrelu Activation 
def leakyrelu_test(z):
    a = np.maximum(0.01*z,z)
    # a =  z * (z > 0)  This is RELU function
    return a

### linear_activation_forward 
def linear_activation_forward_test(A_prev, W, b, activation):
    if activation == "leakyrelu":       
        Z = linear_forward_test(A_prev, W, b)
        A = leakyrelu_test(Z)
    
    elif activation == "linearOutput":
        Z = linear_forward_test(A_prev, W, b)
        A = linearOutput_test(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    return A

### L_model_forward for prediction
def L_model_forward_test(X_test, parameter): 
    A = X_test
    L = len(parameter) // 2  
   
    # Implement [LINEAR -> leaky RELU]*(L-1). 
    for l in range(1, L):
        A_prev = A 
        A = linear_activation_forward_test(A_prev, parameter["W"+str(l)], parameter["b"+str(l)],"leakyrelu")

    # Implement [LINEAR ->linearOutput] at the output layer. 
    AL = linear_activation_forward_test(A, parameter["W"+str(L)], parameter["b"+str(L)], "linearOutput")    
    assert(AL.shape == (4, X_test.shape[1]))        
    return AL

### Predict accuracy and Least Square Error
def predict_accuracy (predict,Y):
    error=np.mean(np.abs(predict-Y)/Y)
    accuracy= (1-error)*100
    LS_error=np.sum(np.power((predict-Y),2))    
    return accuracy, LS_error
    

########################### READ DATA FROM EXCEL and PREPROCESSING DATA ########################
### Read Data from Excel file 
xls = pd.ExcelFile('/Users/esther/source/repos/Python Models/RawData.xlsx')
data_x = pd.read_excel(xls,'Inputs', header=0)
data_x_array = data_x.as_matrix()
data_y = pd.read_excel(xls,'Outputs', header=0)
data_y_array = data_y.as_matrix()

### Splitting and Scaling Data
X_train, X_test, Y_train, Y_test= train_test_split(data_x_array,data_y_array, test_size=0.3)
x_train = meanNormalization(X_train).T
x_test  = meanNormalization(X_test).T
y_train = Y_train.T
y_test  = Y_test.T


########################### DATA VISUALISING ###################################################

# Need to delete 'header=0' to loop through: 'Dimensions_in', 'Flow_rate_in', 'Pressure_in ', 'Power_in', 'Temperature_in'  
"""
def plot_histogram(data_x, cols, bins = 10):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        data_x[col].plot.hist(ax = ax, bins = bins) 
        ax.set_title('Histogram of ' + col) 
        ax.set_xlabel(col) 
        ax.set_ylabel('Number of autos')
        plt.show() 
plot_histogram(data_x, data_x.columns)
"""

############################ MODEL TRAINNING AND TESTING ########################################

parameters = L_layer_model(x_train, y_train, layers_dims = (5, 30, 35,4),learning_rate=0.003, num_iterations = 50000,lambd=0.1, print_cost=True)
predict_y_train = L_model_forward_test(x_train, parameters)
predict_y_test = L_model_forward_test(x_test, parameters)
accuracy_train,LS_error_train = predict_accuracy (predict_y_train,y_train)
accuracy_test, LS_error_test = predict_accuracy(predict_y_test, y_test)
print("Training Set Accuracy: " + str(round(accuracy_train,2))+"%")
print("Training Set Least Square Error: " + str(LS_error_train))
print("Test Set Accuracy: " + str(round(accuracy_test,2))+"%")
print("Test Set Least Square Error: " + str(LS_error_test))


############################## WRITING PARAMETERS TRAINED TO CSV FILE#############################
myData = parameters
with open('C:/Users/esther/source/repos/Python Models/parameters.csv','w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in myData.items():
       writer.writerow([key, value])
print("Writing parameters complete.")


###
