import streamlit as st

st.set_page_config(page_title="IMDB Sentiment Exploration and train", layout="wide")
st.title("IMDB Sentiment Analysis training - Multi-page App")

st.write("""
Welcome! Use the sidebar to navigate:
- **Exploration**: load a CSV, preview, simple stats & charts
- **Training**: choose a ML model, set TF-IDF options, train & optimize (Grid/Randomized)
- **Test**: type a review and predict; test on random dataset samples
- **Deep Learning (optional)**: load a pre-trained Keras LSTM/CNN and try predictions
""")