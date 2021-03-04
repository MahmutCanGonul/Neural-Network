# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:40:37 2019

@author: Mahmut Can Gönül
"""
import numpy as np

xall = np.array(([10,2],[3,8],[5,20],[2,9]),dtype = float) 
y =  np.array(([67],[98],[54]), dtype = float)

xall = xall/np.amax(xall,axis=0)
y = y/100


x = np.split(xall,[3])[0]
xpredict = np.split(xall,[3])[1]

class Neural_Network(object):
    def __init__(self):
        self.inputsize = 2
        self.outputsize = 1
        self.hiddensize = 3
        self.w1 = np.random.randn(self.inputsize,self.hiddensize)    
        self.w2 = np.random.randn(self.hiddensize,self.outputsize) 
        
    def forward(self,x):
        self.z = np.dot(x,self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.w2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    def sigmoidprime(self,s):
        return s * (1-s)
    def backforward(self,x,y,o):
        self.o_error = y-o
        self.o_delta = self.o_error*self.sigmoidprime(o)
        
        self.z2_error =  self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error*self.sigmoidprime(self.z2)
        
        self.w1 = x.T.dot(self.z2_delta) 
        self.w2 = self.z2.T.dot(self.o_delta)
        
    def train(self,x,y,):
        o = self.forward(x)
        self.backforward(x,y,o)
    
    def saveWeights(self):
        np.savetxt("w1 text",self.w1,fmt = "%s")
        np.savetxt("w2 text",self.w2,fmt = "%s")
        
    def predict(self):
        print("Predicted Data based on tranied weights")
        print("Input: \n"+str(xpredict))
        print("Output: \n"+str(self.forward(xpredict)))
    
NN = Neural_Network()
for i in  range(2000):
  print("#"+ str(i)+"\n") 
  print("Input: \n"+str(x))
  print("Output: \n"+str(y))   
  print("Predicted Output: \n"+ str(NN.forward(x)))
  print("Loss: \n"+ str(np.mean(np.square(y - NN.forward(x)))))      
  print("\n")
  NN.train(x,y)



NN.saveWeights()
NN.predict()      
        
        
    
    




        
    
                  
                


