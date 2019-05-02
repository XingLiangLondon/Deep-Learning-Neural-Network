import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv 
%matplotlib inline 

nodeNumber_layer1= 10
nodeNumber_layer2= 6

### For nureal network processing, every inputs and outputs feeding into the nureal netwrok are in the format of (numberoffeatures,numberofsamples)
### Raw data Min-Max Normalized  
def meanNormalization (Xraw):    
    # number of samples 
    m=Xraw.shape[0] 
    #print('m', m)
    # number of features
    n=Xraw.shape[1] 
    #print('n', n)
    #print('Xraw', Xraw)
    u=1/m*Xraw.sum(axis=0)
    #print('u', u)
    u=u.reshape((1,n)) 
    xmax=Xraw.max(axis=0)
    #print('xmax', xmax)
    xmax=xmax.reshape((1,n)) 
    xmin=Xraw.min(axis=0) 
    #print('xmin', xmin)
    xmin=xmin.reshape((1,n))
    Xnorm=(Xraw-u)/(xmax-xmin) 
    #print('xnorm', Xnorm)
    return Xnorm.T

###Read CSV Data
df = pd.read_csv('/Users/liangx/Desktop/test.csv') 
print(df.head())
x_data=(np.asanyarray(df[['time','areas']]))
y_data=(np.asanyarray(df[['output']]))
print("x_data_shape=", x_data.shape)
print("y_data_shape=", y_data.shape)


### Splitting and Scaling Data 
#uppercase X_ Y_ is raw data before nomalisation, lowercase x_ y_ is the processed data after normalisation
X_train, X_test, Y_train, Y_test= train_test_split(x_data,y_data, test_size=0.33)
print("number of trainning samples", X_train.shape[0])
print("number of test samples", X_test.shape[0])
print("X_train", X_train)
print("X_test", X_test)
x_train = meanNormalization(X_train) 
print("x_train.shape", x_train.shape)
print("x_train", x_train)
x_test  = meanNormalization(X_test)
print("x_test.shape", x_test.shape)
print("x_test", x_test)
y_train = Y_train.T
print("y_train.shape", y_train.shape)
print("y_train", y_train)
y_test  = Y_test.T
print("y_test.shape", y_test.shape)
print("y_test", y_test)


###Building Nureal Network
# numFeatures is the number of features in our input data.
numFeatures = x_train.shape[0]
print("number of trainning features", numFeatures)
# numSamples is the number of samples.
numOutputs = y_train.shape[0]
print("number of trainning outputs", numOutputs)


# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
def create_placeholders(numFeatures, numOutputs):
    
    X = tf.placeholder(tf.float32, [numFeatures, None], name="X") 
    Y = tf.placeholder(tf.float32, [numOutputs, None], name="Y") 
    return X, Y

# Weight Tensor
def initialize_parameters():
    
    W1 = tf.get_variable("W1", [nodeNumber_layer1, numFeatures],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # Bias tensor
    b1 = tf.get_variable("b1", [nodeNumber_layer1, 1],initializer= tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [nodeNumber_layer2, nodeNumber_layer1],initializer=tf.contrib.layers.xavier_initializer (seed=1))
    # Bias tensor
    b2 = tf.get_variable("b2", [nodeNumber_layer2, 1],initializer= tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [numOutputs, nodeNumber_layer2],initializer=tf.contrib.layers.xavier_initializer (seed=1))
    # Bias tensor
    b3 = tf.get_variable("b3", [numOutputs, 1],initializer= tf.zeros_initializer())
    parameters ={"W1":W1,
                 "b1":b1,
                 "W2":W2,
                 "b2":b2,
                 "W3":W3,
                 "b3":b3
                }
    return parameters
"""
# Debugging code for def initialize_parameters(): 
tf.reset_default_graph()
with tf.Session()as sess:
    parameters= initialize_parameters()
    print("W1= "+str(parameters["W1"]))
    print("b1= "+str(parameters["b1"]))
    print("W2= "+str(parameters["W2"]))
    print("b2= "+str(parameters["b2"]))    
    print("W3= "+str(parameters["W3"]))
    print("b3= "+str(parameters["b3"]))
"""
def forward_propagation(X, parameters):
    W1= parameters['W1']
    b1= parameters['b1']
    W2= parameters['W2']
    b2= parameters['b2']
    W3= parameters['W3']
    b3= parameters['b3']
    Z1= tf.add(tf.matmul(W1, X),b1)
    #print("W1= "+str(W1))   
    #print("X= "+str(X))
    #print("tf.matmul(W1, X)=", tf.matmul(W1, X))
    #print("b1= "+str(b1))
    #print("Z1= "+str(Z1))
    A1=tf.nn.leaky_relu(Z1, alpha=0.01, name='Leaky_ReLU_layer1')
    #print("A1= "+str(A1))
    Z2= tf.add(tf.matmul(W2,A1),b2)
    A2=tf.nn.leaky_relu(Z2, alpha=0.01, name='Leaky_ReLU_layer2')
    Z3= tf.add(tf.matmul(W3, A2),b3) #Layer3 is pure liner i.e. Z3=A3
    return Z3

"""
# Debugging code for def forward_propagation(X, parameters):
tf.reset_default_graph()
with tf.Session()as sess:
    X = tf.placeholder(tf.float32, [numFeatures, None], name="X") 
    Y = tf.placeholder(tf.float32, [numOutputs, None], name="Y") 
    parameters= initialize_parameters()
    Z3= forward_propagation(X, parameters)
    print("Z3= "+str(Z3))
"""

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.square(tf.subtract(Y, Z3)))
    return cost

"""
# Debugging code for def compute_cost(Z3, Y): 
tf.reset_default_graph()
with tf.Session()as sess:
    X = tf.placeholder(tf.float32, [numFeatures, None], name="X") 
    Y = tf.placeholder(tf.float32, [numOutputs, None], name="Y") 
    parameters= initialize_parameters()
    Z3= forward_propagation(X, parameters)
    cost= compute_cost(Z3, Y)
    print("cost= "+str(cost))
"""



def model (X_train, Y_train, X_test, Y_test, Learning_rate=0.0001, num_iterations=15000, print_cost=True):
    tf.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y =  create_placeholders(numFeatures, numOutputs)


    # Initialize parameters
    parameters =  initialize_parameters()
 
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
  
    
    # Cost function: Add cost function to tensorflow graph
    cost =compute_cost(Z3, Y)
  
    # Backward propagation: Define the tensorflow optimizer. Use a GradientDescentOptimizer or an AdamOptimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = Learning_rate).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate = Learning_rate).minimize(cost)
    
    
    #Run SGD
    
    init= tf.global_variables_initializer() 
    with tf.Session() as sess:
        
        sess.run(init)
        #Do the training looping
        for i in range(num_iterations):
            
            
            _,per_interation_cost = sess.run([optimizer,cost], feed_dict={X: x_train, Y: y_train})
                       
            # Print the cost every epoch
            if print_cost == True and i % 100 == 0:
                print ("Cost after interarion %i: %f" % (i, per_interation_cost))
            if print_cost == True and i % 5 == 0:
                costs.append(per_interation_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(Learning_rate))
        plt.show()
        train_accuracy = np.mean(np.argmax(y_train, axis=1) ==sess.run([optimizer,cost], feed_dict={X: x_train, Y: y_train}))
        test_accuracy  = np.mean(np.argmax(y_test, axis=1) ==sess.run([optimizer,cost], feed_dict={X: x_test, Y: y_test}))
        print("interations= %d, train accuracy = %.2f%%, test accuracy = %.2f%%"% (i + 1, 100. * train_accuracy, 100. * test_accuracy))

                
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
        print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
        return parameters

parameters= model(x_train, y_train, x_test, y_test)


"""
#Cost Function: It is a function that is used to minimize the difference between the right answers (labels) and estimated outputs by our Network.
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Type of optimization: Gradient Descent
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    
#Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

sess.close() #finish the session
"""
