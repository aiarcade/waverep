#Program will take set of audio file create a new audio stream by adding a group of repetitions
#Then append both files and apply a feature extraction and treat it as a Xset for training data.
#Random numbers used for repetitions are considered as Y set

from os import listdir
from os.path import isfile, join
import random
import numpy as np
from pydub import AudioSegment
import array
import multiprocessing
import pandas as pd
import librosa



#maximum length of the input audio file
MAX_LENGTH=25000000
#Source file for dataset is available in this directory
IN_PATH="source/"
#Generated dataset files will be stored in this directory
OUT_PATH="train/"
#Maximum number of  random numbers . 

NO_RANDOMS_MAX=4
#Number of files handled per process 
NFILES_PER_PROCESS=4

#Take a  mp3 file input and generate a row in the datset and return it
class GenXyset(object):
    def __init__(self,input_wav,output_wav,no_randoms,length):
        if input_wav.find(".wav") >0:
            self.in_wav=AudioSegment.from_wav(input_wav)
        else :
            self.in_wav=AudioSegment.from_mp3(input_wav)
        self.output_wav=output_wav
        self.no_randoms=no_randoms
        self.length=length
        self.input=input_wav
    def process(self):
        print("processing ",self.input,"with randoms",self.no_randoms)
        r = random.sample(range(2, 10), self.no_randoms)
        r.sort()
        s = random.sample(range(3,7), self.no_randoms)
        s.sort()
        start_point=0
        end_point=0
        sp=[]
        ep=[]
        y=""
        for i in range(0,len(s)):
            end_point=len(self.in_wav)/r[i]
            if start_point==0:
                self.out_wav=self.in_wav[start_point:end_point]
            else:
                self.out_wav=self.out_wav+self.in_wav[start_point-(start_point/s[i-1]):end_point]
            start_point=end_point
            #Binary encoding of Y.
            y=y+(bin(s[i]).replace("0b","")).zfill(4)+(bin(r[i]).replace("0b","")).zfill(4)        

        self.out_wav=self.out_wav+self.in_wav[end_point-(end_point/s[i]):]
        combined=self.in_wav+self.out_wav
        raw = np.array(combined.get_array_of_samples())
        padded= np.zeros(self.length,dtype=float)
        padded[:raw.shape[0]] = raw
        y_row=self.input+","+self.output_wav
        for i in range(0,len(s)):
            y_row=y_row+","+str(r[i])+","+str(s[i])
        if len(s)<4:
            for i in range(0,4-len(s)):
                y_row=y_row+",0,"+"0"
        stft=librosa.feature.chroma_stft(y=padded, sr=self.in_wav.frame_rate)
        return y_row,stft
            
# st a batch files and call the GenXyset to process it one by one
# Return a list of rows for dataset  
class GenXysetOnBatch(object):
    def __init__(self,batch_files):      
         self.batch=batch_files
    def run(self):
        print("Running batch with",self.batch )
        xyset=[]
        for _file in self.batch:
            in_file=IN_PATH+_file
            for i in range(1,NO_RANDOMS_MAX+1):
                out_file=OUT_PATH+"train-r"+str(i)+"-"+_file.replace("mp3","wav")
                gd=GenXyset(in_file,out_file,i,MAX_LENGTH)
                xyset.append(gd.process())
        return xyset

#helper function to split a list of filenames into chunks of filenames
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
#process function to run the batch processing by calling   GenXysetOnBatch 
# Return the dataset rows    
def batch_run(batch):
    batch_gen=GenXysetOnBatch(batch)
    xyset=batch_gen.run()
    return xyset

if __name__ == "__main__":  
    
    
    files = [f for f in listdir(IN_PATH) if isfile(join(IN_PATH, f))]
    files_lst=[]
    for file_name in files:
        files_lst.append(file_name)
            
    input_batch=chunks(files_lst,NFILES_PER_PROCESS)
    #write y set header
    outyfile=open(OUT_PATH+"yt.txt","w")
    outyfile.write("inputfile,trainxfile,r0,s0,r1,s1,r2,s2,r3,s3\n")
    #create a multiprocessing pool to speed up the dataset generation
    p = multiprocessing.Pool()
    first_complete=0
    for xyset in p.imap(batch_run,input_batch):
        for xy in xyset:
            outyfile.write(xy[0]+"\n")
            x=xy[1].reshape(1,xy[1].shape[0]*xy[1].shape[1])
            if first_complete==0:
                X=x.copy()
                first_complete=1
            else:
                X=np.vstack((X,x))
    outyfile.close()
    Xdata=pd.DataFrame(X)
    #write the X set to hd5 file
    Xdata.to_hdf('train/Xdata.h5', key='df', mode='w')
