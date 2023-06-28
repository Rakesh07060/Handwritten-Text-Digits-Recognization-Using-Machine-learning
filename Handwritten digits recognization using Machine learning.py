#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install tensorflow


# In[2]:


import tensorflow as tf


# In[4]:


mnist=tf.keras.datasets.mnist


# In[8]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[9]:


x_train.shape


# In[13]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0],cmap=plt.cm.binary)


# In[14]:


print(x_train[0])


# In[17]:


#train the data set
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)


# In[18]:


print(x_train[0])


# In[19]:


print(y_train[0])


# In[25]:


import numpy as np
IMG_SIZE=28
x_trainr=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_testr=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(x_trainr.shape,x_testr.shape)


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D


# In[31]:


model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(32))
model.add(Activation("relu"))


model.add(Dense(10))
model.add(Activation("softmax"))


# In[33]:


model.summary()


# In[34]:


print(len(x_trainr))


# In[35]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[36]:


model.fit(x_trainr,y_train,epochs=5,validation_split=0.3)


# In[37]:


prediction=model.predict(x_testr)


# In[38]:


print(prediction)


# In[41]:


print(np.argmax(prediction[2]))


# In[43]:


plt.imshow(x_test[2])


# In[44]:


import cv2


# In[76]:


img = cv2.imread("C:\\Users\\rakesh\\Desktop\\vfx.PNG")


# In[77]:


plt.imshow(img)


# In[78]:


img.shape


# In[79]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[80]:


gray.shape


# In[81]:


resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)


# In[82]:


resized.shape


# In[83]:


cng=tf.keras.utils.normalize(resized,axis=1)


# In[84]:


new=np.array(cng).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# In[85]:


new.shape


# In[86]:


predictions=model.predict(new)


# In[87]:


print(np.argmax(predictions))


# In[ ]:




