import numpy as np 
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
		self.x = x
		self.y = y
		self.lr = lr #learning rate
		self.innerlayer = 4;
		self.outerlayer = 1;
		self.W_one = np.random.rand(self.innerlayer, 6)
		self.W_two = np.random.rand(self.outerlayer, self.innerlayer)

	def  backProp(self):

		for i in range (5000):

			self.z_two = np.dot(self.W_one, self.x)
			self.a_two = sigmoid(self.z_two)
			self.z_three = np.dot(self.W_two,self.a_two)
			self.a_three = sigmoid(self.z_three)

			self.delta_three = np.multiply((self.a_three - self.y), sigmoid_prime(self.z_three)) 
			print(self.delta_three)
			self.pot = np.multiply(self.delta_three, sigmoid_prime(self.z_two))
			self.delta_two =  np.dot(self.W_two.T, self.pot)

			self.dEdW_one = np.dot(self.delta_two, self.x.T)
			self.dEdW_two = np.dot(self.delta_three, self.a_two.T)

			self.W_one = self.W_one - (self.lr * self.dEdW_one)
			self.W_two = self.W_two - (self.lr * self.dEdW_two)

		return self.a_three

NN = Neural_Net(x_minmax_scaled, y_minmax_scaled, 0.1)

y_hat = NN.backProp()

print(y_hat)


