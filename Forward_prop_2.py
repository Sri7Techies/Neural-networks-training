import numpy as np 
from numpy import exp, array, random, dot
import pandas as pd 
from sklearn import preprocessing

'''
	Project demonstrating simple forward propagation
	in Neural networks for a little complex data.
	Data used is the marks of three exams
	which are used to predict final score
	Later we shall use the same data for backpropagation algo
'''

min_max_scaler = preprocessing.MinMaxScaler()

'''
	Min_max_scaler is available in sklear. It scales data in the range of [0,1]
	example: A is a list of values (array)
	for a given index  i
	min_max_scaler(A[i]) = (A[i] - min(A[i]))/(max(A[i]) - min(A[i]))
	This works well because our data is non-negative
'''

tests_score = np.array([[78,85,91],[65,78,51],[69,80,75],[96,94,91],[86,90,78],[68,76,65]])

final_score = np.array([[88],[71],[85],[95],[80],[60]])
  
#data preperation

x_scaled = preprocessing.scale(tests_score)
y_scaled = preprocessing.scale(final_score)

x_minmax_scaled = min_max_scaler.fit_transform(x_scaled)
y_minmax_scaled = min_max_scaler.fit_transform(y_scaled)




#helper functions

#sigmoid
def sigmoid(i):
	return (1 / (1 + np.exp(-i)))

#derivative of signoid ==> g' = g * (1 - g) where g is sigmoid
def sigmoid_prime(i):
	return (sigmoid(i)) * (1 - sigmoid(-i))

#actual neural network

class Neural_Net(object):

	def __init__(self, x, y, lr):

		random.seed(1)

		self.x = x
		self.y = y
		self.lr = lr #learning rate
		self.innerlayer = 3 #input layer
		self.hiddenlayer = 10
		self.outerlayer = 1
		self.W_one = 2 * np.random.rand(self.hiddenlayer, self.innerlayer) - 1
		self.W_two = 2 * np.random.rand(self.outerlayer, self.hiddenlayer) - 1

	def  forwardProp(self):
		#	Z = W.X
		self.z_two = np.dot( self.W_one, self.x) #(10,6) matrix
		#	A = σ(Z)
		self.a_two = sigmoid(self.z_two)
		self.z_three = np.dot(self.W_two, self.a_two)#(1,6) matrix
		self.a_three = sigmoid(self.z_three)

		return 100 * self.a_three

NN = Neural_Net(x_minmax_scaled.T, y_minmax_scaled.T, 0.01)

y_hat = NN.forwardProp()

print(y_hat.T)


