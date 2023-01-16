import streamlit as st
import keras
import tensorflow as tf
from math import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

input_shape=(75,75,3)
inputs = Input(shape=input_shape)
xception = tf.keras.applications.xception.Xception(weights="imagenet",include_top=False)(inputs)
x = keras.layers.GlobalAveragePooling2D(name='model-global-pool-1')(xception)
flatten = Flatten(name='model-flatten')(x)
gender_model = Dense(128,activation='relu',name='gender-dense-1')(flatten)
gender_model = keras.layers.Dense(1, activation="sigmoid",name='gender-output') (gender_model)
age_model = Dense(128,activation='relu',name='age-dense-1')(flatten)
age_model = keras.layers.Dense(1, activation="relu",name='age-output') (age_model)
model = Model(inputs=inputs, outputs=[gender_model,age_model])
model.load_weights('best_model.h5')

st.markdown("""<p style="font-size:50px;font-weight:bold"> Age & Gender Prediction App </p> """,True)
try:
    file = st.file_uploader('Upload image file')
    if(file.name):
        img_path = file.name
        st.success("Successful file upload!!")
except:
    print('Waiting for upload')
    
st.markdown("""<p style="font-size:30px;font-weight:bold"> Prediction Results </p> """,True)
try:
    img = cv2.imread(img_path)
    img = cv2.resize(img,(75,75),interpolation = cv2.INTER_LINEAR)
    img = np.array(img)
    gender_dict = {0:'Male',1:'Female'}
    pred = model.predict(img.reshape(1,75,75,3))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    st.text('Predicted Gender is '+pred_gender)
    st.text('Predicted Age is '+str(pred_age))
    st.image(img_path)
    plt.axis('off');
except:
    print(0)