#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import math
def tanh(n):
    return np.tanh(n)


# In[3]:


z = np.array([[5,-1],[2,-1],[3.1,-1],])    #Input
print("the input vector dimension is",z.shape)


# In[4]:


d = np.array([[0],[1],[1],[0],]) 
print("teacher values", d)


# In[9]:


print("Weights of the hidden layer")
v = np.array([[1,2,3],[2,6,5]])                                 #Initiate v matrix
print("value of v:\n",v)
print("the dimension of the weights is",v.shape)


# In[10]:


print("Weights ")
w = np.random.rand(7,1)                                 #Initiate weight w
print("Initial Weight:\n",w)


# In[11]:


neta=1                                                    #Fix neta value
iteration=10000                                           #fix the epochs value


# In[21]:


for i in range(1,iteration):
    for j,n in enumerate(z):
        #print("----Iteration----",i)
        y_net = np.dot(z[j],v)                           #calculate y_net
        #print("Value of y_net:\n",y_net)
        y = tanh(y_net)                               #calculate y value
        y = np.append(y,[1])
        y = y.reshape(7,1)
        #print("Value of y:\n",y)
        wt = np.transpose(w)                              #Transpose weight matrix
        #print("Transpose shape:\n",wt.shape)
        out_net = np.dot(wt,y)                             #calculate out_net value
        out = tanh(out_net)
        #print("out value:\n",out)
        del_o =(d[j]-out)*(1-out)*out                      #calculate del_o Unipolar function
        #print("delta out:",del_o)
        #err=np.sum(del_o*w)
        del_hid = del_o*((y)*(1-y))*w                      #Hidden Layer
        #print("del_hid value:\n",del_hid)
        w=w+neta*del_o*y                                   #Update weight
        #print("weight value:\n",w)
        del_hid1=np.delete(del_hid,-1)
        del_hid1=del_hid1.reshape(2,1)
        #print("reshape to calculate::\n",del_hid1)
        v=v+neta*del_hid1*z[j]                             #update v value and print
        #print("The v value:\n",v)
print("Final w",w)
print("Final v",v)


# In[ ]:




