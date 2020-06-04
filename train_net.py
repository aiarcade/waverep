import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf



y=pd.read_csv('train/y.txt')
X=pd.read_hdf('train/Xdata.h5'
, 'df').values
y=y.loc[:, 'r0':].values

input_dim = X.shape[1]
nb_out = y.shape[1]

print(input_dim)






model = Sequential()
model.add(Dense(1000, input_dim = input_dim))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(nb_out))
#model.add(Activation('linear'))
optimizer = RMSprop(0.001)

model.compile(loss='mean_absolute_error', optimizer="adam",metrics=['mae', 'mse'])


X=tf.keras.utils.normalize(X, axis=-1, order=2)

model.fit(X, y, epochs=100, batch_size=10, validation_split=0.1, verbose=2)
model.save_weights('model_weights.h5')
model.save('model_net.h5')
p=model.predict(X)





    
    
