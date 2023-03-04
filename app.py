import numpy as np
import streamlit as st
from PIL import Image as im
from streamlit_option_menu import option_menu
import pickle
import tensorflow as tf

@st.cache
def predict(image):
    img = tf.image.decode_jpeg(image.read(), channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    pred = cnn.predict(img)
    return pred

load_model=pickle.load(open('train_model.sav','rb'))

st. set_page_config(layout="wide")

selected = option_menu(
           menu_title=None,  # required
           options=["Home", "Classification", "Group 13"],  # required
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
if selected == "Classification":
    st.title('Image Classifier')
    st.write('Upload an image to classify')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    with st.sidebar:
                selected = option_menu(
                    menu_title="Classification",  # required
                    options=["BloodMinst", "BreastMinst"],  # required
                    icons=["circle", "square"],  # optional
                    menu_icon="cast",  # optional
                    default_index=0,  # optional
                )
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        prediction = predict(uploaded_file)
        st.write(f"Prediction: {prediction}")

                           
if selected == "Group 13":
    st.markdown('''
                <p style="font-family:Sans serif; color:White; text-align: center;font-size: 50px;">
                Created by : <br>
                     Radha Pradeep Kumar / (K2238531)<br>
                     Debasish Mishra / (K2245963)<br>
                     Hala Nejad Fazel / (K2243949)<br>
                     Umme Salma Kapadia / (K2248066)<br>
                </p>''',
                unsafe_allow_html=True
                ) 
    


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











