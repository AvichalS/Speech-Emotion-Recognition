# Speech-Emotion-Recognition-System
Developed a deep learning model using Multi-Layer Perceptron to recognize and classify speech signals into 6 distinct emotions. Extracted 160 audio features from every audio in a diverse dataset of around 7500 tracks by over 90 actors, enabling the model to detect emotions with around 75% accuracy on the training set. Implemented the model on a user-friendly Streamlit dashboard, allowing seamless audio input and real-time emotion analysis.

The project follows the following steps:

1. Audio Feature Extraction: The raw audio files are preprocessed to extract meaningful features such as Mel-Frequency Cepstral Coefficients (MFCCs), which are commonly used for speech signal processing.
2. Hyperparameter Tuning: The model is trained with various hyperparameters and the best set of hyperparameters is selected using cross-validation.
3. Deep Learning using MLP: The model architecture is an MLP neural network with multiple hidden layers. The network is trained using the selected hyperparameters.
4. Testing on Sample Files: The trained model is tested on sample speech files to evaluate its performance.
5. Building a Dashboard: Finally, a dashboard is built to allow users to record audio and classify emotions using the trained model.
