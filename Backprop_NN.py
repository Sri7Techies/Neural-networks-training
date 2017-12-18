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

class Neural_Net(object):
	#initialize hyper-parameters (ite is the number of iterations of gradiant descent)
	def __init__(self, x, y, lr, ite):
        self.x = x
        self.y = y
        self.lr = lr
        self.ite = ite
        self.innerlayer = 3 #input layer
        self.hiddenlayer = 7
        self.outerlayer = 1
        self.W_one = 2 * np.random.rand(self.hiddenlayer, self.innerlayer) - 1
        self.W_two = 2 * np.random.rand(self.outerlayer, self.hiddenlayer) - 1
    
    def backProp(self):
        
        for i in range(self.ite):
            #basic forward propogation stff
            self.z_1 = np.dot( self.W_one,self.x)
            self.a_1 = sigma(self.z_1)
            self.z_2 = np.dot(self.W_two, self.a_1)
            self.a_2 = self.sigma(self.z_2)
	
            #calculation of error and deltas for each layer
            self.error = (self.a_2 - self.y)
            self.delta_2 = self.error * self.sigma_der(self.z_2)
            self.error_2 = np.dot(self.W_two.T, self.delta_2)
            self.delta_1 = np.multiply(self.error_2, self.sigma_der(self.z_1))
		
            #slope of cost with respective weights
            self.DJDW_2 = np.dot(self.delta_2, self.a_1.T)
            self.DJDW_1 = np.dot(self.delta_1, self.x.T)
	
            #updating weights with new values
            self.W_one -= self.lr * self.DJDW_1
            self.W_two -= self.lr * self.DJDW_2
        
        return 100 * self.a_2

    def sigma(self,a):
        return 1 / (1 + np.exp(-a)) 
    
    def sigma_der(self,a):
        return np.exp(-a)/((1+np.exp(-a))**2)

NN = Neural_Net(x_minmax_scaled.T, y_minmax_scaled.T, 0.05, 10000)

y_hat = NN.backProp()

print(y_hat.T)


