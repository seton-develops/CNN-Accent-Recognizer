# CNN-Accent-Recognizer
English Based Accent Recognizer using Convolutional Neural Network

This project was originally intended for my own personal use. Thus, there are some wrappers that need to be made for
public use. However, in order to use it in its current state. One merely needs to follow the directions below:

1) Make changes to path of where English Accents are stored. I have included Hispanic and "American" accents in this project. By default, I have already listed the path I used.
2) Make changes to path where you want the feature extraction data to be saved to
3) Run preprocess_audio.py. Feature extraction data should be saved to path in step 2
4) Run English Accent Recognizer. Model will begin training. The console should say that you are ready to record audio when it is done.
5) Speak for 30seconds. The accent recognizer should tell you if you are hispanic or not.

