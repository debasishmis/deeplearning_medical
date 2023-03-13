import numpy as np
import base64
import io
import cv2
import streamlit as st
from PIL import Image as im
from streamlit_option_menu import option_menu
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import pickle


filename = "model_bld.sav"
loaded_model1 = pickle.load( open(filename, "rb" ) )

filename = "model_brst.sav"
loaded_model = pickle.load( open(filename, "rb" ) )

def preprocess(image):
    # Convert the image to a PIL image object
    img = tf.cast(image, tf.float32)
    img = image/ 255.0
    # Convert the image to a NumPy array
    img = cv2.resize(img, dsize=(28, 28))
    img = np.array(img).reshape(-1, 28,28,1)
    return img

def preprocess1(image):
    # Convert the image to a PIL image object
    img1 = tf.cast(image, tf.float32)
    img1 = image/ 255.0
    # Convert the image to a NumPy array
    img1 = cv2.resize(img1, dsize=(28, 28))
    img1 = np.array(img1).reshape(1, 28,28,3)
    return img1




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
        <p style="font-family:Optima; color:#3A7355; text-align: center;font-size: 100px;"><strong>
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
            options=["CNN Breastmnist", "Model2","Best Model"],  # required
            icons=["circle", "square","traingle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            
        )
    if selected2== "CNN Breastmnist" : 
        if uploaded_file1 is not None:
            with st.sidebar:
                st.image(uploaded_file1, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file1)
            img = im.open(uploaded_file1)
            img = preprocess(np.asarray(img))
            prediction1 = loaded_model.predict(img)
            if prediction1[0][0] == prediction1.max():
                st.write(f'<p style="color:White;font-size: 50px;"><strong>The Breastmnist Sample is malignant </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
                st.balloons()   
                image = im.open('/Users/debasish/Downloads/Plots/AUC_brst.png')
                st.image(image,use_column_width=50)
         
                image = im.open('//Users/debasish/Downloads/Plots/accuracy_brst.png')
                st.image(image,use_column_width=50)

                image = im.open('/Users/debasish/Downloads/Plots/Loss_brst.png')
                st.image(image)
            elif prediction1[0][1] == prediction1.max():
                st.write(f'<p style="color:White;font-size: 50px;"><strong>The Breastmnist Sample is normal, benign </strong></p>', 
                         unsafe_allow_html=True)
                st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction1.max()} </strong></p>', 
                         unsafe_allow_html=True)
                st.balloons()   
                image = im.open('plots/AUC_brst.png')
                st.image(image,use_column_width=50)
         
                image = im.open('plots/accuracy_brst.png')
                st.image(image,use_column_width=50)

                image = im.open('plots/Loss_brst.png')
                st.image(image)
            else:
                st.write(f'<p style="color:White;font-size: 50px;"><strong>The Breastmnist Sample is not a proper sample </strong></p>', 
                         unsafe_allow_html=True)
                
        else:
            st.warning('Please upload an image')
            
        

if selected == "Bloodmnist":
    st.title('Image Classifier')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    add_selectbox = st.sidebar.selectbox("Model Selection:", 
                                        ("CNN Bloodmnist", "Model2","Best Model","Please select model"),
                                        index=3
                                        )
    if add_selectbox == 'CNN Bloodmnist':
        with st.sidebar:
            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                img = im.open(uploaded_file)
            else:
                st.warning('Please upload an image')
        img = im.open(uploaded_file)
        img = preprocess1(np.asarray(img))
        prediction = loaded_model1.predict(img)
        if prediction[0][0] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is basophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][1] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is eosinophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][2] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is erythroblast </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][3] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is immature granulocytes(myelocytes, metamyelocytes and promyelocytes) </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][4] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is lymphocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][5] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is monocyte </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][6] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is neutrophil </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        elif prediction[0][7] == prediction.max():
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is platelet </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction.max()} </strong></p>', 
                     unsafe_allow_html=True)
        else:
            st.write(f'<p style="color:White;font-size: 50px;"><strong>The Bloodmnist Sample is not a proper sample </strong></p>', 
                     unsafe_allow_html=True)
            st.write(f'<p style="color:White;font-size: 50px;"><strong>Prediction: {prediction}</strong></p>', 
                     unsafe_allow_html=True)
        st.balloons()   
        col1,col2=st.columns(2)
        with col1:
            image = im.open('plots/AUC_bld.jpg')
            st.image(image)
        with col2:
            image = im.open('plots/Loss_accuracy_bld.png')
            st.image(image)
    elif add_selectbox == 'Please select model':
        st.warning('No option is selected')
    else:
        st.warning('Please upload an image')
        
if selected == "Credits":
    st.markdown('''
        <p style="font-family:Sans serif; color:White; text-align: center;font-size: 50px;">
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
        background-image: url('https://c1.wallpaperflare.com/preview/602/232/972/woman-person-desktop-work.jpg'); 
        background-repeat: no-repeat;
        background-size: cover;
    }
    
    [data-testid="stHeader"]{
        background-color:rgba(0,0,0,0);
        }
    
    </style>
    """,
    unsafe_allow_html=True
)
