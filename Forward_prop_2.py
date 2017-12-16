import numpy as np 
from numpy import exp, array, random, dot
import pandas as pd 
from sklearn import preprocessing

"""
	Project demonstrating simple backpropogation
	in Neural networks.
	Data used is the marks of three exams
	which are used to predict final score

"""

min_max_scaler = preprocessing.MinMaxScaler()

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
		self.W_one = 2 * np.random.rand(self.innerlayer, self.hiddenlayer) - 1
		self.W_two = 2 * np.random.rand(self.hiddenlayer, self.outerlayer) - 1

	def  forwardProp(self):

		

		self.z_two = np.dot( self.x, self.W_one)
		self.a_two = sigmoid(self.z_two)
		self.z_three = np.dot(self.a_two, self.W_two)
		self.a_three = sigmoid(self.z_three)

		return self.a_three

NN = Neural_Net(x_minmax_scaled, y_minmax_scaled, 0.1)

y_hat = NN.forwardProp()

print(100 * y_hat)


