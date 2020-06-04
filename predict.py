import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf 
from pydub import AudioSegment
import array
import multiprocessing
import pandas as pd
import librosa
import sys

#adjust it corresponding to generate_net
MAX_LENGTH=25000000

file1 = sys.argv[1] #input audio 
file2 = sys.argv[2] #input audio that is created from splitjoin


if file1.find(".wav") >0:
    f1_wav=AudioSegment.from_wav(file1)
else :
    f1_wav=AudioSegment.from_mp3(file1)
if file2.find(".wav") >0:
    f2_wav=AudioSegment.from_wav(file2)
else :
    f2_wav=AudioSegment.from_mp3(file2)

x1=np.array(f1_wav.get_array_of_samples())
x2=np.array(f2_wav.get_array_of_samples())

X=np.zeros(MAX_LENGTH,dtype=float)
X[:x1.shape[0]] = x1
X[x1.shape[0]+1:x1.shape[0]+1+x2.shape[0]]=x2
stft=librosa.feature.chroma_stft(y=X, sr=f1_wav.frame_rate)
X_p=stft.reshape(1,stft.shape[0]*stft.shape[1])
X_p=tf.keras.utils.normalize(X_p, axis=-1, order=2)
model=load_model('model_net.h5')
y=model.predict(X_p)
print("Output order :r0 s0 r1 s1 r2 s2 r3 s3 ")
print("Randoms corresponding to the input")
print(y)








    
    
