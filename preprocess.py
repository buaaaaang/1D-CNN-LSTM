import csv
import wave
from pathlib import Path
import pygame
import time
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py

path = "C:/Users/LG/Desktop/skt/KoreanSpeechDataForSER"
emotion = ["Angry","Disgust","Fear","Neutral","Sadness"] 
#we will noet consider happiness and surprise (there are only few samples of those emotions)
len_audio = 4
sr = 16000
n_total = 14605 + 10011 - 121 - 56 - 322 - 124

def get_duration(audio_path):
    audio = wave.open(audio_path)
    frames = audio.getnframes()
    rate = audio.getframerate()
    duration = frames / float(rate)
    return duration


def listen(audio_path):
    pygame.mixer.init()
    S = pygame.mixer.Sound(audio_path)
    S.set_volume(1)
    S.play()
    time.sleep(get_duration(audio_path))

def preprocess(filepath):
    with h5py.File(filepath,'w') as h:
        x_total = h.create_dataset('x',(n_total,sr*len_audio),dtype='float32')
        y_total = h.create_dataset('y',(n_total,),dtype='int8')
        
        index = np.array(range(len(x_total)))
        np.random.shuffle(index)

        f = open(path+'/4th.csv','r')
        rdr = csv.reader(f)
        next(rdr)
        i = 0
        for line in rdr:
            if line[3] in emotion:
                x, _ = librosa.load(path+"/4th/"+line[0]+".wav", sr=sr)
                if len(x) < sr*len_audio:
                    pad = sr*len_audio-len(x)
                    x = np.pad(x,(pad//2,pad-pad//2),'constant',constant_values=0)                
                else: x = x[:sr*len_audio]
                x_total[index[i]] = (np.float32(x))
                y_total[index[i]] = (np.float32(emotion.index(line[3])))
                i += 1
        f.close()

        f = open(path+'/5th.csv','r')
        rdr = csv.reader(f)
        next(rdr)
        for line in rdr:
            if line[3] in emotion:
                x, _ = librosa.load(path+"/5th/"+line[0]+".wav", sr=sr)
                if len(x) < sr*len_audio:
                    pad = sr*len_audio-len(x)
                    x = np.pad(x,(pad//2,pad-pad//2),'constant',constant_values=0)                
                else: x = x[:sr*len_audio]
                x_total[index[i]] = (np.float32(x))
                y_total[index[i]] = (emotion.index(line[3]))
                i += 1
        f.close()


if __name__=="__main__":
##############################################################################
    '''
    #checking audiofile
    f = open(path+'/4th.csv','r')
    rdr = csv.reader(f)
    next(rdr)
    count = 0
    lineNum = 2
    line = next(rdr)
    minDuration = get_duration(path+"/4th/"+line[0]+".wav")
    maxDuration = get_duration(path+"/4th/"+line[0]+".wav")

    y, sr = librosa.load(path+"/4th/"+line[0]+".wav",sr=48000)
    print(sr)
    time = np.linspace(0,len(y)/sr,len(y))
    fig, p = plt.subplots()
    p.plot(time,y,label='speech waveform')
    plt.show()
    
    for line in rdr:
        lineNum += 1
        duration = get_duration(path+"/4th/"+line[0]+".wav")
        if minDuration>duration: minDuration=duration
        if maxDuration<duration: maxDuration=duration
        if (duration < 8): count += 1
    f.close()

    print(minDuration)
    print(maxDuration)
    print(count)
    '''
##############################################################################
    '''
    #checking number of each emotions
    f = open(path+'/4th.csv','r')
    rdr = csv.reader(f)
    count = [0,0,0,0,0]
    next(rdr)
    for line in rdr:
        if (line[3] in emotion): count[emotion.index(line[3])] += 1
    f.close()
    print(count)
    '''
##############################################################################  
    #checking if preprocessing works
    #preprocess(path + "/1D.hdf5")
    from keras.utils.io_utils import HDF5Matrix

    with h5py.File(path+"/1d.hdf5","r") as f:
        print(len(f['x'][0]))
    
    x_tr = HDF5Matrix(path+"/1d.hdf5",'x',end=16000)
    print(x_tr.shape[1:])


