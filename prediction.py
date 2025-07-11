import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences

model=load_model('models/lstm_model.keras')
vectorizer_model=load_model("models/vectorizer.keras")
vectorizer=vectorizer_model.layers[0]

def predict_next_word(model,vectorizer,text,max_sequence_len):
    token_list=vectorizer(tf.constant([text])).numpy()[0]
    if len(token_list) >=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    vocab=vectorizer.get_vocabulary()
    word_index={word:i for i,word in enumerate(vocab)}
    for word,index in word_index.items():
        if index==predicted_word_index[0]:
            return word
    return None

input_text="To be or not to be"
print(f"Input Text: {input_text}")
max_sequence_len=model.input_shape[1]+1
next_word=predict_next_word(model,vectorizer,input_text,max_sequence_len)
print(f"Next word Prediction: {next_word}")

