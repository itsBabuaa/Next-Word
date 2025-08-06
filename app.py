import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model_LSTM= load_model('model-LSTM.h5')

# Load the GRU model
model_GRU= load_model('model-GRU.h5')

# Load the tokenizer
tokenizer= pickle.load(open('tokenizer.pkl', 'rb'))

# Fn to predict the next word

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]
    
    token_list = pad_sequences([token_list], maxlen = max_sequence_len)
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# Streamlit UI

st.title("📋 Next Word Prediction")

input_text = st.text_input("Enter the sequence of words")

col1, col2= st.columns(2)

with col1:
    if st.button("Predict Next word With LSTM"):
        
        if not input_text:
            st.warning("⚠️ Please enter a somthing before clicking Predict Next word!")
        else:
            max_sequence_len = model_LSTM.input_shape[1]+1
            next_word = predict_next_word(
                model=model_LSTM, 
                tokenizer=tokenizer, 
                text=input_text, 
                max_sequence_len=max_sequence_len,
            )
            st.header(f"Next Word: {next_word}")

with col2:
    if st.button("Predict Next word With GRU"):
        
        if not input_text:
            st.warning("⚠️ Please enter a somthing before clicking Predict Next word!")
        else:
            max_sequence_len = model_GRU.input_shape[1]+1
            next_word = predict_next_word(
                model=model_GRU, 
                tokenizer=tokenizer, 
                text=input_text, 
                max_sequence_len=max_sequence_len,
            )
            st.header(f"Next Word: {next_word}")