'''
Created on May 13, 2020
Code borrowed Velero Velardo's Music Genre Classification Tutorial
https://www.youtube.com/watch?v=dOG-HxpbMSw&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=16

Data is from International Dialects of English Archive
This code a CNN to recognize hispanic and American speakers for English.
One can test its accuracy with microphone input.
@author: Sean Tonthat
'''
import tensorflow.keras as keras
import pydub
import math
import time
import random
import librosa
import speech_recognition as sr
import json
import numpy as np
from sklearn.model_selection import train_test_split


def load_json(path):
    with open(path, "r") as read:
        data = json.load(read)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data["mapping"])
    return x,y,z

def build_model(input_shape):
    model = keras.Sequential()
    #keras.layers.Conv2D(number of kernals, kernal size, actuvation function)
    
    #convultional layer 1
    model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides =(2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization())

    #convultional layer 2
    model.add(keras.layers.Conv2D(32, (2,2), activation = "relu", input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides =(2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization())
    
    #flatten into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu")) #64 neurons
    model.add(keras.layers.Dropout(0.3)) #avoids overfitting
    
    #output layer
    model.add(keras.layers.Dense(2, activation="softmax")) #two different categories
    return model

def prepare_datasets(test_size,valid_size):
    #x = sample vectors
    #y = labels
    #Z = semantic labels
    x,y,z = load_json("C:\\Users\\Sean\\eclipse-workspace\\Accent-Recog\\data.json")
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train)
    
    #convert into a 4D array
    x_train = x_train[...,np.newaxis]
    x_valid = x_valid[...,np.newaxis]
    x_test = x_test[...,np.newaxis]
    return x_train, x_valid, x_test, y_train, y_valid, y_test, z

#gets micophone data for testing
def get_input(categories):
    print("\nTESTING WITH NOVEL DATA")
    print("\nPress a key to start recording...")
    temp = input()
    
    time_dur = 0
    while time_dur < 30:
        print("Please speak for 30 seconds. Cannot be of shorter length")
        start = time.time()
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            with open('sample.wav','wb') as file:
                file.write(audio.get_wav_data())
        time_dur = time.time() - start
        if time_dur < 30:
            print("try again. insuffienct length")
    print("finished recording")
    
    print("input the number corresponding your accent")
    for i, category in enumerate(categories):
        print(category, "is ", i)
    test_label = input()
    return test_label        

#trims file to 30seconds   
def trim_input(filepath):
    thirty_sec = 30 * 1000
    audio = pydub.AudioSegment.from_wav(filepath)
    audio = audio[:thirty_sec]
    audio.export("sample.wav", "wav")
    
#gets mfcc vector for test audio input
def get_mfcc(file_path):
    data = {
        "mfcc": []
    }
    
    n_mfcc=13
    n_fft = 2048
    hop_length =512
    num_segments = 10
    duration = 30
    sample_rate = 22050

    SEGMENTS_PER_FILE = sample_rate *duration      
    num_samples_per_seg = int(SEGMENTS_PER_FILE/num_segments)
    num_MFCC_vectors_per_seg = math.ceil(num_samples_per_seg / hop_length)   

    signal,sr = librosa.load(file_path, sr =sample_rate)
    for s in range(num_segments):
        start_sample = num_samples_per_seg * s
        finish_sample = start_sample + num_samples_per_seg
        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr = sr, n_fft = n_fft, n_mfcc = n_mfcc, hop_length = hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_MFCC_vectors_per_seg:
            data["mfcc"].append(mfcc.tolist())
            
    with open("data2.json", "w") as fp:
        json.dump(data, fp, indent = 4)

def predict(model, sample, label, categories):
    sample = sample[np.newaxis,...]
    prediction = model.predict(sample)
    predicted_index = np.argmax(prediction, axis= 1)

    print("EXPECTED ", label)
 
    print("PREDICTED", predicted_index)
    

if __name__ == "__main__":
    x_train, x_valid, x_test, y_train, y_valid, y_test, categories = prepare_datasets(0.25,0.2)
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss ="sparse_categorical_crossentropy", metrics = ['accuracy'])
    
    model.fit(x_train, y_train, validation_data = (x_valid, y_valid), batch_size = 32, epochs = 30)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose =1)
    print("accuracy on test set is: {}".format(test_accuracy))

    expected_label = get_input(categories)
    trim_input("C:\\Users\\Sean\\eclipse-workspace\\Accent-Recog\\sample.wav")
    get_mfcc("C:\\Users\\Sean\\eclipse-workspace\\Accent-Recog\\sample.wav")
     
    with open("C:\\Users\\Sean\\eclipse-workspace\\Accent-Recog\\data2.json", "r") as read:
        data = json.load(read)
    sample_x = np.array(data["mfcc"])
    sample_x = sample_x[...,np.newaxis]
    num = random.randint(1,10)
    predict(model, sample_x[num], expected_label, categories)
    
    model.save("English_Accent_Recog.h5")
    