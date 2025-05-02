import streamlit as st
import numpy as np
import librosa
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading
import time
from tensorflow import keras
from tensorflow.keras.models import load_model

# Set page config at the very beginning
st.set_page_config(page_title="Dysarthria Speech Classifier", page_icon="🎤", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://wallpapercave.com/wp/wp4697667.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp .block-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        font-weight: bold;
    }
    /* Style for expander sections */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 10px 10px 0 0;
    }
    .streamlit-expanderContent {
        background-color: white !important;
        border-radius: 0 0 10px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the custom H5 model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('C:/Users/athar/Downloads/dysarthia.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Function to preprocess audio for the model
def preprocess_audio(audio_data, sample_rate):
    # Implement your preprocessing steps here
    # This is just a placeholder - adjust according to your model's requirements
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Function to analyze audio using the loaded model
def analyze_audio(audio_data, sample_rate):
    # Preprocess the audio
    processed_audio = preprocess_audio(audio_data, sample_rate)
    
    # Reshape the input as required by your model
    model_input = processed_audio.reshape(1, -1)  # Adjust this based on your model's input shape
    
    # Make prediction
    prediction = model.predict(model_input)
    
    # Interpret the prediction
    # Assuming the model outputs a single value between 0 and 1
    # where values closer to 1 indicate dysarthria
    dysarthria_probability = prediction[0][0]
    
    # Classification based on a threshold (you may need to adjust this)
    threshold = 0.5
    is_dysarthria = dysarthria_probability < threshold
    
    return "Dysarthria Detected" if is_dysarthria else "No Dysarthria Detected", dysarthria_probability

def display_results(prediction, probability):
    st.header("Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification")
        st.write(f"**Prediction:** {prediction}")
       # st.write(f"**Confidence:** {probability:.2%}")
        
    with col2:
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        labels = ['Dysarthria', 'No Dysarthria']
        sizes = [1 - probability, probability]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Update the animated character with results
    character_col1, character_col2, character_col3 = st.columns([1, 2, 1])
   # with character_col2:
    #    if "Dysarthria" in prediction:
     #       st.info("Dysarthria Detected", icon="🗨️")
      #  else:
       #     st.success("No Dysarthria Detected", icon="🗨️")
    st.markdown(
        f"""
        
            <div>
                {"🗨️ " + prediction}
            </div>
        
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("Dysarthria Speech Classifier")
    st.write("**Analyze speech patterns to detect dysarthria characteristics**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Speech Analysis")
        tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

        with tab1:
            st.write("Upload an audio file (.wav, .mp3) of speech to analyze for dysarthria characteristics.")
            uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Display audio player
                st.audio(tmp_file_path, format="audio/wav")

                if st.button("Analyze File"):
                    with st.spinner("Analyzing..."):
                        audio_array, sample_rate = librosa.load(tmp_file_path, sr=None)
                        prediction, probability = analyze_audio(audio_array, sample_rate)
                        display_results(prediction, probability)

                # Clean up the temporary file
                os.unlink(tmp_file_path)

        with tab2:
            st.write("Record your voice for 5 seconds, then the model will predict.")
            
            status_placeholder = st.empty()
            result_placeholder = st.empty()

            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": False, "audio": True},
            )

            if webrtc_ctx.audio_receiver:
                status_placeholder.info("Recording... (5 seconds)")
                audio_buffer = []
                start_time = time.time()

                while time.time() - start_time < 5:
                    try:
                        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                        for audio_frame in audio_frames:
                            sound = audio_frame.to_ndarray()
                            audio_buffer.append(sound)
                    except queue.Empty:
                        continue

                status_placeholder.success("Recording complete!")
                
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0)
                    
                    # Save as temporary WAV file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        wavfile.write(tmp_file.name, 22050, audio_data)  
                    
                    # Analyze the recorded audio
                    with st.spinner("Analyzing..."):
                        audio_array, sample_rate = librosa.load(tmp_file.name, sr=22050)
                        prediction, probability = analyze_audio(audio_array, sample_rate)
                        result_placeholder.empty()
                        display_results(prediction, probability)

                else:
                    status_placeholder.warning("No audio recorded. Please try again.")

    #with col2:
     #   st.header("Information")
      #  with st.expander("About Dysarthria", expanded=True):
       #     st.write("Dysarthria is a motor speech disorder that affects the muscles used for speech. "
        #             "People with dysarthria may have slurred, slow, or difficult-to-understand speech.")
        
        #with st.expander("Our Classifier", expanded=True):
         #   st.write("Not sure if you have Dysarthria? Our binary classifier detects the presence or absence of dysarthria in speech samples. "
          #           "It analyzes various acoustic features to make this determination.")

    # Display the animated character
    st.image("C:/Users/athar/Downloads/animated_image.webp", width = 100)

if __name__ == "__main__":
    main()