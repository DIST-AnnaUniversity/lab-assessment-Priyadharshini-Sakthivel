#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import numpy as np
tensorflow.version
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN


# In[5]:


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def reshape_training(vect_seed, dim, sample_size):
    T = len(vect_seed)
    vect_train = np.array([vect_seed]*dim)
    vect_train = np.repeat(vect_train, sample_size)
    vect_train = np.reshape(vect_train, (sample_size, T, dim), order = 'F')
    # vect_train[0]
    return(vect_train)


# In[6]:


sample_size = 256
dim_in = 3
dim_out = 2
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]
x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)


# In[7]:


model=Sequential()
model.add(SimpleRNN(input_shape=(None, dim_in), 
                    return_sequences=True, 
                    units=5))
model.add(SimpleRNN(input_shape=(None,4), 
                    return_sequences=True, 
                    units=7))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 100, batch_size = 32)


# In[8]:


model.get_weights()
model.get_weights()[0].shape # (3,5)
model.get_weights()[1].shape # (5,5)
model.get_weights()[2].shape # (5,1)
model.get_weights()[3].shape # (5,7)
model.get_weights()[4].shape # (7,7)
model.get_weights()[5].shape # (7,1)
model.get_weights()[6].shape # (7,2)
model.get_weights()[7].shape # (2,1)


# In[15]:


W_h = model.get_weights()[0] # W_h a (3,5) matrix
U_h = model.get_weights()[1] # U_h a (5,5) matrix
b_h = model.get_weights()[2] # b_h a (5,1) vector
W_hp = model.get_weights()[3] # W_hp a (5,7) matrix
U_hp = model.get_weights()[4] # U_hp a (7,7) matrix
b_hp = model.get_weights()[5] # b_hp a (7,1) matrix

## For the hidden --> output layer
W_y = model.get_weights()[6] # W_y a (7,2) matrix
b_y = model.get_weights()[7] # b_y a (2,1) vector

# We initialize the hidden vectors with 0
h_0 = np.array([0, 0, 0, 0, 0]) # h_0 a (5,1) vector
hp_0 = np.array([0, 0, 0, 0, 0, 0, 0]) # hp_0 a (7,1) vector
hp0=np.array([1,2,3,4,5,6])#hp0 a(6,1) vector


# In[20]:


new_input = [[4,2,1], [1,1,1], [1,1,1],[2,3,1]]
print(model.predict(np.array([new_input])))


# In[19]:


x_1 = np.array(new_input[0]) # x_1 a (3,1) vector
h_1_before_tanh = np.dot(x_1, W_h) + np.dot(h_0, U_h) + b_h
h_1 = [math.tanh(x) for x in h_1_before_tanh]

# We apply formula (A') at t=1
hp_1_before_tanh = np.dot(h_1, W_hp) + np.dot(hp_0, U_hp) + b_hp
hp_1 = [math.tanh(x) for x in hp_1_before_tanh]

# We apply formula (B) at t=1
y_1_before_sigmoid = np.dot(hp_1, W_y) + b_y
y_1 = [sigmoid(x) for x in y_1_before_sigmoid]
print(y_1) 


# In[12]:


x_2 = np.array(new_input[1]) # x_2 a (3,1) vector
h_2_before_tanh = np.dot(x_2, W_h) + np.dot(h_1, U_h) + b_h
h_2 = [math.tanh(x) for x in h_2_before_tanh]

# We apply formula (A') at t=2
hp_2_before_tanh = np.dot(h_2, W_hp) + np.dot(hp_1, U_hp) + b_hp
hp_2 = [math.tanh(x) for x in hp_2_before_tanh]

# We apply formula (B) at t=2
y_2_before_sigmoid = np.dot(hp_2, W_y) + b_y
y_2 = [sigmoid(x) for x in y_2_before_sigmoid]
print(y_2)


# In[13]:


x_3 = np.array(new_input[2]) # x_3 a (3,1) vector
h_3_before_tanh = np.dot(x_3, W_h) + np.dot(h_2, U_h) + b_h
h_3 = [math.tanh(x) for x in h_3_before_tanh]

# We apply formula (A') at t=3
hp_3_before_tanh = np.dot(h_3, W_hp) + np.dot(hp_2, U_hp) + b_hp
hp_3 = [math.tanh(x) for x in hp_3_before_tanh]

# We apply formula (B) at t=3
y_3_before_sigmoid = np.dot(hp_3, W_y) + b_y
y_3 = [sigmoid(x) for x in y_3_before_sigmoid]
print(y_3)


# In[ ]:




