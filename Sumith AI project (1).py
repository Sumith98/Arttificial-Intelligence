
# coding: utf-8

# In[ ]:


import Librosa
import pandas as pd
import numpy as np




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler




import keras




import warnings
warnings.filterwarnings('ignore')





d=pd.read_csv('prj_data.csv')




d.head()




d=d.drop(['filename'],axis=1)




d.dtypes




d.isnull().sum()





types_list = d.iloc[:, -1]
onehot_le = LabelEncoder()
p = onehot_le.fit_transform(types_list)




p




d.head()





di_std = StandardScaler()




r = di_std.fit_transform(np.array(d.iloc[:, :-1], dtype = float))





X_train, X_test, y_train, y_test = train_test_split(r, p, test_size=0.25)





X_train[10]



 




from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))




model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)




test_loss, test_acc = model.evaluate(X_test,y_test)




print('test_acc: ',test_acc)





x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]






model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)





results



 



predictions = model.predict(X_test)



predictions[0].shape





np.sum(predictions[0])




np.argmax(predictions[0])




