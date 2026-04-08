import streamlit  as  st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model.h5")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

max_len = 6

# def predict(text,top_k=5):
#     #text = lower.text()
#     text = text.lower()
    
#     tokens = tokenizer.texts_to_sequences([text])[0]
#     tokens = pad_sequences([tokens],maxlen  = max_len,padding="pre")

#     preds = model.predict(tokens,verbose=0)[0]
#     #top_indices - preds.argsort()[-top_k:][::-1]
#     top_indices = preds.argsort()[-top_k:][::-1]
#     result = []
#     for idx in top_indices:
#         for word, i  in tokenizer.word_index.items():
#             if  i == idx:
#                 result.append(word)
#                 break
#     return result
def predict(text, top_k=5):
    text = text.lower()

    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = pad_sequences([tokens], maxlen=max_len, padding='pre')

    preds = model.predict(tokens, verbose=0)[0]

    top_indices = preds.argsort()[-top_k:][::-1]

    result = []
    for idx in top_indices:
        for word, i in tokenizer.word_index.items():
            if i == idx:
                if word != "<OOV>":   # 🔥 FILTER
                    result.append(word)
                break

    return result
    st.title("Smart Next Word Predictor")

text = st.text_input("Enter alteast 6 words :")

if st.button("Predict"):
    if text:
        words = predict(text)

        st.write("### Top Predictions :")
        for w in words:
            st.write(" ",w)