## Data collection
import nltk
import pickle
#nltk.download('gutenberg')

from nltk.corpus import gutenberg
import pandas as pd

'''
## load the dataset
data=gutenberg.raw('shakespeare-hamlet.txt')






## save to a file
with open('data/hamlet.txt','w') as file:
    file.write(data)'''


## data preprocessing

import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization,Input
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

## load the dataset
with open('data/hamlet.txt','r') as file:
    text=file.read().lower()

text_ds=tf.data.Dataset.from_tensor_slices([text])
vectorizer=TextVectorization(
    max_tokens=10000,
    output_mode='int',
    standardize='lower_and_strip_punctuation',
    split='whitespace'
)

vectorizer.adapt(text_ds)

vocab=vectorizer.get_vocabulary()
word_index={word:i for i,word in enumerate(vocab)}
total_words=len(vocab)
print(word_index)
print('Vocabulary Size: ',total_words)
print('first 10 tokens:',vocab[:10])

## create input sequences
input_sequences=[]

for line in text.split('\n'):
    tokenized=vectorizer(tf.constant([line])).numpy()[0]
    token_list=[token for token in tokenized if token !=0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print(input_sequences[0])

## pad_sequences
max_sequence_len=max([len(x) for x in input_sequences])

print(max_sequence_len)

input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

print(input_sequences)
print(len(input_sequences))
print(max_sequence_len)

## create predictors and labels 
x,y=input_sequences[:,:-1],input_sequences[:,-1]
from keras.utils import to_categorical
y=to_categorical(y,num_classes=total_words)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

## train our lstm rnn

from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout

model=Sequential()
model.add(Input(shape=(max_sequence_len-1,)))
model.add(Embedding(total_words,output_dim=100))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation='softmax'))

## compile the model
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=35,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
## training of model

history=model.fit(x_train,y_train,epochs=75,validation_data=(x_test,y_test),verbose=1)

model.save("models/lstm_model.keras")

loss,accuracy=model.evaluate(x_test,y_test)
print("accuracy",accuracy)  ## 54.77 % accurate

vectorizer_model=Sequential([vectorizer])
vectorizer_model(tf.constant(["hello world"]))
vectorizer_model.save("models/vectorizer.keras")

