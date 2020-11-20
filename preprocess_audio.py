'''
Created on 5/17/20
Code borrowed Velero Velardo's Music Genre Classification Tutorial
https://www.youtube.com/watch?v=dOG-HxpbMSw&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=16

Data is from International Dialects of English Archive
This code performs MFCC on wav files of accented speakers.
@author: Sean Tonthat
'''

import numpy as np
import librosa
import wave, os, glob
import math
import json

wav_files = []
path = "C:\\Users\\Sean\\Music\\accent_files"
json_path = "data.json"


duration = 30
sample_rate = 22050
SEGMENTS_PER_FILE = sample_rate *duration


def save_mfcc(data_path, json_path, n_mfcc=13, n_fft = 2048, hop_length =512, num_segments = 10):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels":[]
        }
    num_samples_per_seg = int(SEGMENTS_PER_FILE/num_segments)
    num_MFCC_vectors_per_seg = math.ceil(num_samples_per_seg / hop_length) 
    
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(data_path)): #get the audio files from individual folder
        if dirpath is not data_path:
            dirpath_categories = dirpath.split("/")
            semantic_label = dirpath_categories[-1]
            data["mapping"].append(semantic_label)
            
            for f in filenames:
                file_path = os.path.join(dirpath,f)
                signal,sr = librosa.load(file_path, sr =sample_rate)
                print(file_path)
                 
                for s in range(num_segments):
                    start_sample = num_samples_per_seg * s
                    finish_sample = start_sample + num_samples_per_seg
                     
                    #store mfcc if it has expected length
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr = sr, n_fft = n_fft, n_mfcc = n_mfcc, hop_length = hop_length)
                    mfcc = mfcc.T
                    if len(mfcc) == num_MFCC_vectors_per_seg:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1) #create label based on folder
  
                           
    with open(json_path, "w") as fp: #put into json file
        json.dump(data, fp, indent = 4)
                            
#     
# if __name__ == "__main__":
#     save_mfcc(path, json_path)
