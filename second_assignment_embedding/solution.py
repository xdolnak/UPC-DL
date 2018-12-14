# Imports
import numpy as np
import re
import collections
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from keras.layers import Embedding
import pandas as pd

# Parameters
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 200  # Number of epochs we usually start to train with
BATCH_SIZE = 64  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 30 # Maximum number of words in a sequence
GLOVE_DIM = 300  # Number of dimensions of the GloVe word embeddings
root = Path('./')
input_path = root / 'input/'
ouput_path = root / 'output/'
source_path = root / 'source/'

# Preparation 
df = pd.read_csv(input_path / 'Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
df = df[['text', 'airline_sentiment']] 
X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)

# Tokenize text
tk = Tokenizer(num_words=NB_WORDS,
filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
tk.fit_on_texts(X_train) # tokenizer train
tk.fit_on_texts(X_test) # tokenizer test
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

# We set all sequences to an equal size
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# Change sentiment classes to numeric values
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

saved_model = models.Sequential()
saved_model.add(layers.Embedding(NB_WORDS, 100, input_length=MAX_LEN))
saved_model.add(layers.Flatten())
saved_model.add(layers.Dense(3, activation='softmax'))
saved_model.load_weights("./emb_model.h5")

# Network architecture
emb_model = models.Sequential()
emb_model.add(layers.Embedding(NB_WORDS, 100, input_length=MAX_LEN, weights=saved_model.layers[0].get_weights(), trainable=False ))
#emb_model.add(layers.GRU(128, return_sequences=True))
emb_model.add(layers.GRU(128, return_sequences=True))
emb_model.add(layers.GRU(128))
#emb_model.add(layers.Dense(256, activation='relu'))
emb_model.add(layers.Dense(1024, activation='relu'))
#emb_model.add(layers.Flatten())
emb_model.add(layers.Dense(3, activation='softmax'))

#Compile model
emb_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

print(emb_model.summary(), 'summary')

#Train model
emb_history = emb_model.fit(X_train_seq_trunc, y_train_oh, batch_size=BATCH_SIZE, nb_epoch=NB_START_EPOCHS, validation_split=0.1)

#Evaluate the model with test set
score = emb_model.evaluate(X_test_seq_trunc, y_test_oh, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(emb_history.history['acc'])
plt.plot(emb_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.close()
#Loss plot
plt.plot(emb_history.history['loss'])
plt.plot(emb_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_loss.png')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
Y_pred = emb_model.predict(X_test_seq_trunc)
y_pred = np.argmax(Y_pred, axis=1)

#Plot statistics
print('Analysis of results')
target_names = ['Negative', 'Neutral', 'Positive']
print(classification_report(np.argmax(y_test_oh,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test_oh,axis=1), y_pred))

#emb_model.save_weights("emb_model.h5")

'''

# Glove wiki embeddings
glove_file = 'glove.6B.' + str(GLOVE_DIM) + 'd.txt'
emb_dict = {}
glove = open(source_path / glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()

# Store embeddings to matrix NB_WORDS x GLOVE_DIM
emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))
for w, i in tk.word_index.items():
	if i < NB_WORDS:
		vect = emb_dict.get(w)
		if vect is not None:
			emb_matrix[i] = vect
	else:
		break

# Define the same network as for computed embeddings
glove_model = models.Sequential()
glove_model.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
#glove_model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(None, 24, 300) ))
#glove_model.add(layers.MaxPooling1D(pool_size=5, strides=1))
glove_model.add(layers.Flatten())
#glove_model.add(layers.GRU(128, return_sequences=True))
#glove_model.add(layers.GRU(128, return_sequences=True))
#glove_model.add(layers.GRU(128))
glove_model.add(layers.Dense(1024, activation='relu'))
glove_model.add(layers.Dense(3, activation='softmax'))

# Set values and disable learning
glove_model.layers[0].set_weights([emb_matrix]) # Set weights of the embedding layer to the glove values 
glove_model.layers[0].trainable = False # Do not let network to modify these values

#Compile model
glove_model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

print(glove_model.summary(), 'summary')

#Train model
glove_history = glove_model.fit(X_train_seq_trunc, y_train_oh, batch_size=BATCH_SIZE, nb_epoch=NB_START_EPOCHS, validation_split=0.1)

#Evaluate the model with test set
glove_score = glove_model.evaluate(X_test_seq_trunc, y_test_oh, verbose=0)
print('test loss:', glove_score[0])
print('test accuracy:', glove_score[1])

##Store glove Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(glove_history.history['acc'])
plt.plot(glove_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
plt.legend(['train','val'], loc='upper left')
plt.savefig('glove_model_accuracy.png')
plt.close()
#Loss plot
plt.plot(glove_history.history['loss'])
plt.plot(glove_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('glove_model_loss.png')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
Y_pred = glove_model.predict(X_test_seq_trunc)
y_pred = np.argmax(Y_pred, axis=1)

#Plot statistics
print('Analysis of results')
target_names = ['Negative', 'Neutral', 'Positive']
print(classification_report(np.argmax(y_test_oh,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test_oh,axis=1), y_pred))

'''
