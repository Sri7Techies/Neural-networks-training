import numpy as np 

'''
	A simple neural network demonstration using python
	This neural network has one hidden layer with 3 nodes in it
	This demonstration is about simple forward propogation 
'''

#y is the marks scored and x is (sleep, study) hours
y = np.array([[85], [70], [78], [94]])  #to make it 1 x 4 

x = np.array([(8,5),(5,7),(6,4),(8,8)]) #(sleep,study)

#simple forward propogation 
print x

#Sigmoid function to calculate activation values at inner node
def sigmoid(t):
	return (1 / (1 + np.exp(t)))

class Neural_Net(object):

	def __init__(self,x,y):

		self.x = x
		self.y = y
		self.innerlayer = 3
		self.outerlayer = 1

	
	def simplePred(self):

		#simple forward NN
		
		self.w_one = np.random.rand(2, self.innerlayer)
		self.w_two = np.random.rand(self.innerlayer, self.outerlayer)
		self.z_two = np.dot(self.x, self.w_one)
		self.a_two = sigmoid(self.z_two)
		self.z_three = np.dot(self.a_two, self.w_two)
		self.a_three = sigmoid(self.z_three)

		self.y_hat = 100 * self.a_three



		return self.y_hat

	def errorEstimate(self):
		self.error = np.subtract(self.y,self.y_hat)
		print(self.error)
	

rand = Neural_Net(x,y)

y_hat = rand.simplePred()

print("The predicted values of marks using simple forward propogation NN are: ")
print(y_hat)

print("\nThis is rougly the error :")
rand.errorEstimate()
