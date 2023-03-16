import numpy as np
import base64
import io
import cv2
import streamlit as st
from PIL import Image as im
from streamlit_option_menu import option_menu
import joblib
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical
import joblib
import os
import requests





# load model
cnn_model_brst = joblib.load('cnn_brst.joblib)
cnn_bld_model = joblib.load('cnn_bld.joblib)
resnet_brst_model = load_model('resnet_brst.h5)
resnet_bld_model = joblib.load('resnet_bld.joblib)
mobnet_bld_model = load_model('mobnet_bld.h5')



def preprocess(image):
    # Convert the image to a PIL image object
    img = tf.cast(image, tf.float32)
    img = image/ 255.0
    # Convert the image to a NumPy array
    img = cv2.resize(img, dsize=(28, 28))
    img = np.array(img).reshape(-1, 28,28,1)
    return img

def preprocess_resnet(image):
    # Convert the image to a PIL image object
    img = image.astype('float32') / 255.0
    # Convert the image to a NumPy array
    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    img = cv2.resize(img, dsize=(224, 224))
    img = np.array(img).reshape(-1, 224,224,3)
    #img = tf.image.resize(img, dsize=(224, 224))
    return img
@st.cache_resource
def preprocess_resnet_bld(image):
    # Convert the image to a PIL image object
    img = image.astype('float32') / 255.0
    # Convert the image to a NumPy array
    img = tf.image.resize(img, size=(64, 64))
    img = np.array(img).reshape(-1, 64,64,3)
    return img
@st.cache_resource
def preprocess_mobnet_bld(image):
    # Convert the image to a PIL image object
    img = image.astype('float32') / 255.0
    # Convert the image to a NumPy array
    img = tf.image.resize(img, size=(32, 32))
    img = np.array(img).reshape(-1, 32,32,3)
    return img
@st.cache_resource
def preprocess_cnn_bld(image):
    # Convert the image to a PIL image object
    img = tf.cast(image, tf.float32)
    img = image/ 255.0
    # Convert the image to a NumPy array
    img = cv2.resize(img, dsize=(28, 28))
    img = np.array(img).reshape(-1, 28,28,3)
    return img


st.set_page_config(layout="wide")

selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Bloodmnist","Breastmnist", "Credits"],  # required
    icons=["house", "book", "envelope"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "white", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "green"},
    },
)

if selected == "Home":
    st.markdown('''
        <p style="font-family:Optima; color:#B35814; text-align: center;font-size: 100px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>
          Medical Image Analysis and Classification
        </strong></p>''',
        unsafe_allow_html=True
    )  



        
if selected == "Breastmnist":
    st.title('Image Classifier')
    uploaded_file1 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    with st.sidebar:
        selected2 = option_menu(
            menu_title="Model Selection",  # required
            options=["CNN Model", "ResNet", "Best Model"],  # required
            icons=["circle", "square","traingle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
        )
        
    if selected2== "CNN Model" : 
        if uploaded_file1 is not None:
            with st.sidebar:
                st.image(uploaded_file1, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file1)
            img = im.open(uploaded_file1)
            img = preprocess(np.asarray(img))
            prediction1 = cnn_model_brst.predict(img)
            if prediction1[0][0] == prediction1.max():
                st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is malignant </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
            elif prediction1[0][1] == prediction1.max():
                st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is normal, benign </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
            else:
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is not a proper sample </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1}</strong></p>', 
                         unsafe_allow_html=True)
            
            col1,col2,col3=st.columns(3)
            with col1:
                st.metric(label="Precision", value="0.76")   
            with col2:
                st.metric(label="Recall", value="0.56") 
            with col3:
                st.metric(label="Accuracy", value="0.75")
            col1,col2=st.columns(2)
            with col1:
                image = im.open(os.path.join(os.path.abspath(base_path),'accuracy_cnn_brst.png'))
                st.image(image)
            with col2:
                image = im.open(os.path.join(os.path.abspath(base_path),'loss_cnn_brst.png'))
                st.image(image)
            
            image = im.open(os.path.join(os.path.abspath(base_path),'AUC_cnn_brst.png'))
            st.image(image,use_column_width=50)
    if selected2 in ("ResNet","Best Model") : 
        if uploaded_file1 is not None:
            with st.sidebar:
                st.image(uploaded_file1, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file1)
            img = im.open(uploaded_file1)
            img = preprocess_resnet(np.asarray(img))
            prediction1 = resnet_brst_model.predict(img)
            if prediction1 >= 0.5:
                st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is malignant </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
            elif prediction1 < 0.5:
                st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is normal, benign </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
            else:
                st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Breastmnist Sample is not a proper sample </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction1}</strong></p>', 
                         unsafe_allow_html=True)
            
            col1,col2,col3=st.columns(3)
            with col1:
                st.metric(label="Precision", value="0.84")   
            with col2:
                st.metric(label="Recall", value="0.75") 
            with col3:
                st.metric(label="Accuracy", value="0.85")
            
            col1,col2=st.columns(2)
            with col1:
                image = im.open(os.path.join(os.path.abspath(base_path),'accuracy_resnet_brst.png'))
                st.image(image)
            with col2:
                image = im.open(os.path.join(os.path.abspath(base_path),'loss_resnet_brst.png'))
                st.image(image)   
            
            image = im.open(os.path.join(os.path.abspath(base_path),'AUC_resnet_brst.png'))
            st.image(image)   
    else:
        st.warning('Please upload an image')   


if selected == "Bloodmnist":
    st.title('Image Classifier')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    add_selectbox = st.sidebar.selectbox("Model Selection:", 
                                        ("CNN Bloodmnist", "ResNet50","MobileNet","Best Model","Please select model"),
                                        index=4
                                        )
    if add_selectbox in('CNN Bloodmnist','Best Model'):
        with st.sidebar:
            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file)
            else:
                st.warning('Please upload an image')
        img = im.open(uploaded_file)
        img = preprocess_cnn_bld(np.asarray(img))
        prediction = cnn_bld_model.predict(img)
        if prediction[0][0] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is basophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][1] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is eosinophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][2] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is erythroblast </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][3] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is immature granulocytes(myelocytes, metamyelocytes and promyelocytes) </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][4] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is lymphocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][5] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is monocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][6] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is neutrophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][7] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is platelet </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        else:
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is not a proper sample </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction}</strong></p>', 
                     unsafe_allow_html=True)
  
        
            
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric(label="Precision", value="0.91")   
        with col2:
            st.metric(label="Recall", value="0.90") 
        with col3:
            st.metric(label="Accuracy", value="0.91")
            
        col1,col2=st.columns(2)
        with col1:
            image = im.open(os.path.join(os.path.abspath(base_path),'AUC_bld_cnn.png'))
            st.image(image)
        with col2:
            image = im.open(os.path.join(os.path.abspath(base_path),'Loss_accuracy_cnn_bld.png'))
            st.image(image)
            
    elif add_selectbox == 'Please select model':
        st.warning('No option is selected')

    elif add_selectbox == "ResNet50":
        with st.sidebar:
            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file)
            else:
                st.warning('Please upload an image')
        img = im.open(uploaded_file)
        img = preprocess_resnet_bld(np.asarray(img))
        prediction = resnet_bld_model.predict(img)
        if prediction[0][0] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is basophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][1] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is eosinophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][2] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is erythroblast </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][3] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is immature granulocytes(myelocytes, metamyelocytes and promyelocytes) </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][4] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is lymphocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][5] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is monocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][6] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is neutrophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][7] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"text-shadow: 2px 4px 4px rgba(46,91,173,0.6);><strong>The Bloodmnist Sample is platelet </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        else:
            st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is not a proper sample </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction}</strong></p>', 
                     unsafe_allow_html=True)


            
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric(label="Precision", value="0.68")   
        with col2:
            st.metric(label="Recall", value="0.58") 
        with col3:
            st.metric(label="Accuracy", value="0.68")
            
        col1,col2=st.columns(2)
        with col1:
            image = im.open(os.path.join(os.path.abspath(base_path),'accuracy_resnet_bld.png'))
            st.image(image)
        with col2:
            image = im.open(os.path.join(os.path.abspath(base_path),'loss_resnet_bld.png'))
            st.image(image)

        image = im.open(os.path.join(os.path.abspath(base_path),'AUC_resnet_bld.png'))
        st.image(image)
            
    elif add_selectbox in ("MobileNet"):
       with st.sidebar:
           if uploaded_file is not None:
               st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
               img = im.open(uploaded_file)
           else:
               st.warning('Please upload an image')
       img = im.open(uploaded_file)
       img = preprocess_mobnet_bld(np.asarray(img))
       prediction = mobnet_bld_model.predict(img)
       if prediction[0][0] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is basophil </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][1] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is eosinophil </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][2] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is erythroblast </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][3] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is immature granulocytes(myelocytes, metamyelocytes and promyelocytes) </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][4] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is lymphocyte </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][5] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is monocyte </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][6] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is neutrophil </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       elif prediction[0][7] == prediction.max():
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is platelet </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction.max()} </strong></p>', 
                    unsafe_allow_html=True)
       else:
           st.write(f'<p style="color:White;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>The Bloodmnist Sample is not a proper sample </strong></p>', 
                    unsafe_allow_html=True)
           st.write(f'<p style="color:#B35814;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);"><strong>Prediction: {prediction}</strong></p>', 
                    unsafe_allow_html=True)
        
       col1,col2,col3=st.columns(3)
       with col1:
           st.metric(label="Precision", value="0.78")   
       with col2:
           st.metric(label="Recall", value="0.74") 
       with col3:
           st.metric(label="Accuracy", value="0.74")
    
       col1,col2=st.columns(2)
       with col1:
           image = im.open(os.path.join(os.path.abspath(base_path),'accuracy_mobnet_bld.png'))
           st.image(image)
       with col2:
           image = im.open(os.path.join(os.path.abspath(base_path),'loss_mobnet_bld.png'))
           st.image(image)
           
       image = im.open(os.path.join(os.path.abspath(base_path),'AUC_mobnet_bld.png'))
       st.image(image)   
        
         
    else:
        st.warning('Please upload an image')    
  
      
if selected == "Credits":
    st.markdown('''
        <p style="font-family:Sans serif; color:#3A7355; text-align: center;font-size: 50px;text-shadow: 2px 4px 4px rgba(46,91,173,0.6);">
        Created by : <br>
             Group13<br>
             Radha Pradeep Kumar / (K2238531)<br>
             Debasish Mishra / (K2245963)<br>
             Hala Nejad Fazel / (K2243949)<br>
             Umme Salma Kapadia / (K2248066)<br>
        </p>''',
        unsafe_allow_html=True) 
 

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://images.unsplash.com/photo-1599727277757-3f54e54ea618?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1932&q=80'); 
        background-repeat: no-repeat;
        background-size: 100% 100%;
        color : #FFFFFF;

    }
    
    [data-testid="stHeader"]{
        background-color:rgba(0,50,50,0);
        }
    
   div.css-1r6slb0.e1tzin5v2 {
    background-color: #0E1117;
    border: 2px solid #0E1117;
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
    opacity:0.5;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
