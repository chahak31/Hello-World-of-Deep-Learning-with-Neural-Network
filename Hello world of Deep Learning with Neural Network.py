#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[3]:


model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])


# In[4]:


model.compile(optimizer='sgd',loss='mean_squared_error')


# In[5]:


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0],dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0],dtype=float)


# In[6]:


model.fit(xs,ys,epochs=500)


# In[9]:


print(model.predict([20.0]))


# In[ ]:




