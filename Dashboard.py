# DS_Major-Project_Group-1

# To run this script using terminal, use ::  streamlit run 4\ -\ Dashboard.py --server.fileWatcherType none

# Dashboard Source Code

import streamlit as st
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa
import librosa.util, librosa.display
import numpy as np
import mir_eval
import scipy
import seaborn as sns
from IPython.display import Audio
import cmath
import pickle
import tensorflow

# Function for recording audio
def record():
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paInt16,channels=2,rate=44100,
	                input=True,frames_per_buffer=1024)
	frames = []
	for i in range(0, int(44100 / 1024 * 2)): # Record for 2 seconds
	    data = stream.read(1024)
	    frames.append(data)
	stream.stop_stream()
	stream.close()
	p.terminate()
	wf = wave.open("rec.wav", 'wb')
	wf.setnchannels(2)
	wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
	wf.setframerate(44100)
	wf.writeframes(b''.join(frames))
	wf.close()

# Extracting features for saved wav file
def extract_features():
	y,_ = librosa.load('rec.wav'); sr = 22500
	y,_ = librosa.effects.trim(y)
	MFCC = librosa.feature.mfcc(y=y, sr=sr); MFCC = [np.mean(x) for x in MFCC]
	y_harmonic, y_percussive = librosa.effects.hpss(y)
	y_harmonic, y_percussive = y_harmonic.mean(), y_percussive.mean()
	C = librosa.cqt(y); C_mean = [np.mean(x) for x in C]; C_mean = [complex(x).real for x in C_mean]
	chroma = librosa.feature.chroma_cqt(C=C, sr=sr); chroma_mean = [np.mean(x) for x in chroma]
	a, b = [],[]
	for j in range(len(chroma_mean)):
	    polar = cmath.polar(complex(chroma_mean[j])); a.append(polar[0]); b.append(polar[1])
	onset_envelope = librosa.onset.onset_strength(y=y, sr=sr); onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
	onsets = onsets.shape[0]; tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
	c_sync = librosa.util.sync(chroma, beats, aggregate=np.median); c_sync = [np.mean(x) for x in c_sync]
	c, d = [],[]
	for j in range(len(c_sync)):
	    polar = cmath.polar(complex(c_sync[j])); c.append(polar[0]); d.append(polar[1])
	spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr); spectral_bandwidth = spectral_bandwidth.mean()
	spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]; spectral_rolloff = spectral_rolloff.mean()
	spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr); spectral_centroids = spectral_centroids.mean()
	arr = np.hstack(([], MFCC, y_harmonic, y_percussive, C_mean, a, b, onsets, tempo, beats.shape[0], c,d, spectral_bandwidth, spectral_rolloff, spectral_centroids,1,0,0,0,0))
	return y, arr

# Predicting emotion from the features using trained model
def predict_emotion(arr):
	emotions = ['Angry', 'Disgust', 'Fear', 'Happy/Joy', 'Neutral', 'Sad']
	with open('speech_emotion_classifier.pkl','rb') as f:
	    model = pickle.load(f)
	    pred = model.predict(arr.reshape((1,165)))
	    return (emotions[int(np.where( pred == pred.max() )[1])])

# Dashboard Styling
st.set_page_config(page_title='Speech Emotion Recognition', page_icon='🗣️')
st.title('Speech Emotion Recognition')
st.caption('DS Major Project - Group 1')

pressed = st.button('Record Audio')

if pressed:
	record()
	st.write('Recorded your speech.')
	st.audio('rec.wav')

	# Plotting graphs on the dashboard
	y, arr = extract_features()
	fig,ax = plt.subplots(figsize=(15,3))
	plt.subplot(1,3,1)
	librosa.display.waveshow(y=y, sr=22500);
	plt.title('Time Series')
	plt.subplot(1,3,2)
	y_stft = np.abs(librosa.stft(y))
	y_stft = librosa.amplitude_to_db(y_stft, ref=np.max)
	plt.colorbar(librosa.display.specshow(y_stft, x_axis='time', y_axis='log'))
	plt.title('Frequency Domain')
	plt.subplot(1,3,3)
	y_mel = librosa.feature.melspectrogram(y=y, sr=22500)
	y_mel_db = librosa.amplitude_to_db(y_mel, ref=np.max)
	plt.colorbar(librosa.display.specshow(y_mel_db, x_axis='time', y_axis='log'))
	plt.title('Mel Spectrogram')

	st.write('Extracted audio features from your speech.')
	st.pyplot(fig)

	st.write('Predicted Emotion is : ')
	st.title(predict_emotion(arr))
