'''
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import datetime
import numpy as np
import requests
import uuid
import json
import cv2 as cv
from streamlit_webrtc import webrtc_streamer


header = st.container()
instructions = st.container()

input_area = st.container()
sliders = st.container()

with header:
    st.title("Table of Contents Scanner")
    st.text("Add very brief intro")

with instructions:
    st.write("Instructions blah blah blah")
'''



import streamlit as st
from streamlit import session_state
import cv2 as cv
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

def main_body():
    if 'image_captured' not in st.session_state.keys():
        session_state['image_captured'] = None

    camera_result = st.camera_input("Take a level, well lit picture of the table of contents here", key="camera")

    if camera_result:
        session_state['image_captured'] = camera_result
        print("STATE UPDATED :", session_state['image_captured'])

    if session_state['image_captured']:
        img = Image.open(session_state['image_captured'])
        rgb_img = np.array(img)

        bgr_image = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)

        gray = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

        st.image(rgb_img)

        value = streamlit_image_coordinates(rgb_img)

        st.write(value)


        #print(open_cv_img)
        #print(type(open_cv_img))

        #st.image(rgb_img)
        #st.image(gray)

        '''
        Options to crop/define the image space
        https://github.com/andfanilo/streamlit-drawable-canvas
        https://github.com/blackary/streamlit-image-coordinates
        
        '''

main_body()
#    st.write_stream    check this out


#get the webcam thingy working
#get two sliders
#get webcam showing up
#add button for webcam
#make button actually take picture
#display the result picture, cropped around the contour
#ask the user if this result looks good
#figure out how to go back to taking a picture step