
This Program will find repetitions and Jumps in an audio file with reference to another audio file. It uses MLP to predict the positions where repetions occurs. Recommend to run using tf2.2 with GPU .

Requirements:

tensorflow 2.2
pytables
numpy and scipy
pydub 
librosa

File descriptions:

generate_dataset.py -> Generate the Xset and Y for training ANN
train_net.py -> Train the ANN from the generated data and store the model model_net.h5
predit.py -> Predict the random values which indicate the positions of reps from two audio files
split.py -> generate a reps audio file from an input audio 


########## How to exceute ######

Available audio files are stored in thesource directory. If the size of the audio mp3 is bigger than 
3mb , adjust the maximum_length variable in both generate_dataset.py and predit.py  files. This can be achieved by trial and error.
run python3 generate_dataset.py to generate the stft features and store it in Xdata.h5 file in train directory.
this script supports multiple processor in you system to speedup exceution. That can be adjusted by changing NFILES_PER_PROCESS
varaible. Once the dataset generation completed, run train_net.py to start the ANN Training. 
This program strictly uses tensorflow 2.2 or higher . Keras is available within the tensorflow2.2
You requires to tune this script to increase the accuracy. The basic parameters used to train the network is
1) Optimization : adam
2) size : 1000->250->250->250->250->8
3)MSE and MAE as the mertices 
4)default learning rate is 0.01
5)Validation split : 10%
6)L2 norm for data
7)epochs=100, batch_size=10
 Once the training is completed , run the split.py to generate  a test sample . fist argument is the input mp3 file  and
 second argument is the ouput file genearted , third argumet is the name for the waveformat of input file.
 Note the random variables s and r  , [s0,s1,s2,s3] [r0,r1,r2,r3] . 
 Run the predict.py to get the predictions on a file . arguments are orginal wav file  and geneared wav file.


 

