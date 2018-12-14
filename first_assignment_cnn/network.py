import numpy as np
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix

#print keras.__version__

# dimensions of our images.
img_width, img_height = 256, 256

# dirs
train_data_dir = '/home/nct01/nct01003/.keras/datasets/indoor/train'
validation_data_dir = '/home/nct01/nct01003/.keras/datasets/indoor/val'
test_dir = '/home/nct01/nct01003/.keras/datasets/indoor/test'
#params
nb_train_samples = 3360
nb_validation_samples = 640
epochs = 2500	
batch_size = 40

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#network
model = Sequential()
model.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # classifier

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
	      lr=0.0025,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
rescale=1. / 255,
rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.01,
zoom_range=[0.9, 1.25],
horizontal_flip=True,
vertical_flip=False,
fill_mode='reflect')
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=1,
    shuffle=False,
    seed=42,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples // batch_size,
    nb_epoch=epochs,
    validation_data=validation_generator, 
    nb_val_samples=nb_validation_samples)

model.save_weights('model_deep.h5')

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_accuracy_23.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('model_loss_23.pdf')

# Test prediction
test_generator.reset()
pred=model.predict_generator(test_generator, 460)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results_23.csv",index=False)


