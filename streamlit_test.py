"""
pip install streamlit

To run: streamlit run streamlit_test.py 
"""

import streamlit as st
import os
import torch
from PIL import Image
from caption import *
import requests
import tensorflow as tf
from io import BytesIO

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# define several variables that will be required for model prediction
CHECKPOINT_NAME = "coco_checkpoint_resnet_101.tar"
CHECKPOINT_BASE_PATH = os.environ.get("checkpoint_base_path")
WORDMAP_BASE_PATH = os.environ.get("wordmap_base_path")
WORDMAP = WORDMAP_BASE_PATH + "/" + "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
BEAM_SIZE = 5
try:
    st.title("Image Captioning")
    st.write("")
    st.write("")

    checkpoint_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['checkpoint1', 'checkpoint2'], index=0,  key = "model_name")
    checkpoint_name = CHECKPOINT_NAME
    
    model = CHECKPOINT_BASE_PATH+checkpoint_name
    
    radio_label = st.radio('Do you want to load an image via URL?', ['yes', 'no'], index=1)
   
    if radio_label == 'no':
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
            print(image_input)
            st.image(image_input, caption='Uploaded Image.',width = 200 ) # use_column_width=True
            st.write("")
            # call function to generate descriptive captions passing image as an input 
            label = main(image_input, model, WORDMAP, BEAM_SIZE)

            st.write("Generating Descriptive Captions...")
            st.write(label)
    else:
        image_url = st.text_input("Insert an image URL", 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png')
        print(image_url)
        if image_url is not None:
            
            image_input = requests.get(image_url).content
            loaded_image = Image.open(BytesIO(image_input))

            # decode into jpeg image
            image_input = tf.image.decode_jpeg(image_input, channels=3)
            
            # display loaded image to user
            st.image(loaded_image, caption='Loaded Image', width=200)
            
            st.write("")

            # call function to generate descriptive captions passing image as an input 
            label = main(image_input, model, WORDMAP, BEAM_SIZE)

            st.write("Generating Descriptive Captions...")
            st.write(label)


except Exception as e:
    print(e)
    st.write("SOME PROBLEM OCCURED")
