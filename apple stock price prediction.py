#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[218]:


df = pd.read_csv('apple_share_price.csv')


# In[219]:


df.head()


# In[220]:


df.info()


# In[221]:


df.describe()


# In[222]:


df.isnull()


# In[223]:


df.isnull().sum()


# In[244]:


print(df.columns)


# In[246]:


plt.figure(figsize=(15, 8))
plt.title('Stock Prices History')
plt.plot(df['Close']) 
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.show()


# In[226]:



df.rolling(7).mean().head(15)
df['Close'].plot(figsize=(16,6))
df.rolling(window=30).mean()['Open'].plot()


# In[227]:



df1 = df.reset_index()['Close']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1) )
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
df1
    


# In[228]:


train_size = int(len(df1)*0.75)
test_size = len(df1) - train_size
train_size, test_size


# In[232]:


train_data, test_data = df1[0:train_size: ], df1[train_size:len(df1), :1]


# In[233]:


def create_dataset(dataset, time_step = 1):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step),0]
        data_x.append(a)
        data_y.append(dataset[i+time_step,0])
    return numpy.array(data_x), numpy.array(data_y)


# In[235]:


import numpy as np

def create_dataset(dataset, time_step):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i : (i + time_step)]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])
    return np.array(data_x), np.array(data_y)

time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# In[236]:



x_train
x_test


# In[237]:


import numpy as np
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# In[238]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

from keras.layers import Dropout,  Bidirectional


# In[239]:




model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# In[103]:



model.fit(x_train, y_train, epochs=80,validation_data=(x_test, y_test), verbose=1)


# In[247]:


test_predict = model.predict(x_test)
train_predict = model.predict(x_train)


# In[251]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)




# In[252]:


trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[back:len(train_predict)+back, :] = train_predict


# In[266]:



plt.subplots(figsize=(20,10))
plt.plot(trainPredictPlot,color='orange')
plt.plot(testPredictPlot,color='blue')
plt.show()


# In[267]:


# Assuming df1 is your original DataFrame containing dollar signs
plt.subplots(figsize=(20,8))
plt.plot(scaler.inverse_transform(df1), color = 'black')
plt.plot(trainPredictPlot, color = 'orange')
plt.plot(testPredictPlot, color = 'green')
plt.show()

