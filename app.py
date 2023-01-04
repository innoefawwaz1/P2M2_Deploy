import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re


enc_tokenizer = pickle.load(open('eng_tokenizer.pkl','rb'))
dec_tokenizer = pickle.load(open('nld_tokenizer.pkl','rb'))
answer = pickle.load(open('translated_tokenized_decoder.pkl', 'rb'))
inf_enc_model = tf.keras.models.load_model('model_encoder.h5', compile=False)
inf_dec_model = tf.keras.models.load_model('model_decoder.h5', compile=False)

# st.header('Translator')
st.markdown("""
        # English to Dutch Translation
        by : <a href="https://linkedin.com/in/innoefawwaz" target="_blank" >Innoe Fawwaz</a>
        """, unsafe_allow_html=True)
st.markdown('''### Enter English sentences to be translated to Dutch''')
text = st.text_input('Put your text here!')

def clean_text(text):
    text = text.lower()  
    pattern = re.compile('\W')
    text = re.sub(pattern,' ',text).strip()
    text = re.sub('([.,!?()\"])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text

if st.button('submit'):
  input_txt = clean_text(text)
  inf = enc_tokenizer.texts_to_sequences([input_txt])
  inf = pad_sequences( inf , maxlen= 14 , padding='post' )

  state_inf = inf_enc_model.predict(inf,verbose=0)

  word = ''
  sentences = []
  target_seq = np.array([[dec_tokenizer.word_index['start']]])
  while True:
    dec_out, h, c = inf_dec_model.predict([target_seq] + state_inf,verbose=0)

    wd_id = np.argmax(dec_out[0][0])
    word = answer[wd_id]
    sentences.append(word)

    target_seq = np.array([[wd_id]])
    state_inf = [h,c]
    if word == 'end' or len(sentences)>=15:
      
      if sentences[-1] == 'end':
          st.write(' '.join(sentences[:-1]))

      else:
          st.write(' '.join(sentences))
      break
    