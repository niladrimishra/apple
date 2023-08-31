#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd


# In[3]:


df=pd.read_csv('mnist test.csv')


# In[24]:


df.head()


# In[5]:


df.info()


# In[7]:


df.describe()


# In[9]:


train_data=pd.read_csv('mnist test.csv')
train_images = train_data.drop('label', axis=1).values
train_labels = train_data['label'].values
train_images = train_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_data = pd.read_csv('mnist test.csv')
test_images = test_data.values
test_images = test_images / 255.0


# In[10]:



model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[11]:


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[19]:


history = model.fit(train_images.reshape(-1, 784),train_labels,epochs=10,batch_size=128,validation_split=0.2)


# In[20]:


plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[21]:



print(test_images.shape)


# In[22]:


test_images = test_images[:, :784]
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(5, 5, i + 1)
    image = test_images[i][:784].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted Label: {predicted_labels[i]}')
    plt.axis('off')

plt.show()

