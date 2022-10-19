#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
import mlflow
import numpy as np
img_width, img_height = 224, 224


# In[2]:


train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples =400
nb_validation_samples = 100
epochs = 20
batch_size = 16


# In[3]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[4]:


train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,
class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[5]:


saved_model_path=r"Model"
saved_model_path1=r"Model1"
saved_model_path2=r"Model2"


# In[6]:


mlflow.set_tracking_uri("sqlite:///mlflow.db")
reg_model_name = "Fist_Model"
mlflow.set_tracking_uri("http://localhost:5000")
tag=[tf.compat.v1.saved_model.tag_constants.SERVING]
key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


# In[9]:


with mlflow.start_run():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    tf.keras.models.save_model(model,saved_model_path)
    mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path,
                                tf_meta_graph_tags=tag,
                                tf_signature_def_key=key,
                                artifact_path=saved_model_path,
                                registered_model_name=)
    for i in history.history:
        history.history[i]=history.history[i][-1]
    mlflow.log_metrics(history.history)


# In[17]:


from tensorflow.keras.utils import load_img
image = load_img('v_data/test/planes/3.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
if (label[0][0]<0.5) and (label[0][1]<0.5) :
    print("This is NOT a car NOR a plane")
elif label[0][0]<label[0][1]  :
    print("Predicted Class Planes")
else :
    print("Predicted Class Cars")


# In[12]:


reg_model_name1="Second Model"
with mlflow.start_run():
    model1 = Sequential()
    model1.add(Conv2D(32, (2, 2)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(64))
    model1.add(Activation('relu'))
    model1.add(Dense(32))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(2))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model1.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    tf.keras.models.save_model(model,saved_model_path1)
    mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path1,
                                tf_meta_graph_tags=tag,
                                tf_signature_def_key=key,
                                artifact_path=saved_model_path1,
                                registered_model_name=reg_model_name1)
    for i in history.history:
        history.history[i]=history.history[i][-1]
    mlflow.log_metrics(history.history)


# In[19]:


reg_model_name2="Third Model"
with mlflow.start_run():
    model2 = Sequential()
    model2.add(Conv2D(64, (2, 2), input_shape=input_shape))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    model2.add(Conv2D(32, (2, 2)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    model2.add(Conv2D(64, (2, 2)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    model2.add(Flatten())
    model2.add(Dense(64))
    model2.add(Activation('relu'))
    model2.add(Dense(32))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(2))
    model2.add(Activation('softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model2.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    tf.keras.models.save_model(model2,saved_model_path2)
    mlflow.tensorflow.log_model(tf_saved_model_dir=saved_model_path2,
                                tf_meta_graph_tags=tag,
                                tf_signature_def_key=key,
                                artifact_path=saved_model_path2,
                                registered_model_name=reg_model_name2)
    for i in history.history:
        history.history[i]=history.history[i][-1]
    mlflow.log_metrics(history.history)

