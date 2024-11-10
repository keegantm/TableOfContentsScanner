import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
import pandas as pd



def getCameraDataThree():
    camera_result = st.camera_input("Take a level, well lit picture of the table of contents here", key="camera")

    if camera_result:
        
        #THIS IS CAUSING INF LOOP, SINCE MAIN KEEPS RERUNNING AND ALL THE VARS GET RESET
        
        if (st.session_state['image_captured'] != camera_result):
            print("NEW PHOTO RESETTING STATES")
            st.session_state['canvas'] = None
            st.session_state['contour_gathered'] = False
        
        st.session_state['image_captured'] = camera_result

def displayCanvasForEditing():

    img = Image.open(st.session_state['image_captured'])
    rgb_img = np.array(img)

    canvas_result = st_canvas(
        background_image=Image.open(st.session_state['image_captured']),
        drawing_mode='polygon',
        height = rgb_img.shape[0],
        width = rgb_img.shape[1],
        fill_color="",
        stroke_width=5,
        display_toolbar=False,
        update_streamlit=False,
    )

    if canvas_result.json_data is not None:
        
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        if len(objects) == 1 and  st.session_state['contour_gathered'] == False:
            #as soon as one polygon is drawn, we do not want the user to draw more. 
            st.session_state['canvas'] = canvas_result.json_data
            print("SAVED CANVAS DATA, of ONE CONTOUR")
            st.session_state['contour_gathered'] = True
            st.rerun()


def displayCanvasResults():
    
    objects = pd.json_normalize(st.session_state['canvas']["objects"]) # need to convert obj to str because PyArrow
    print("SHOWING DATA FOR CONTOUR")
    print(objects)

    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    #TODO, make this display the image with their polygon on it
    st.dataframe(objects)
    
    reset_canvas_btn = st.button("RESET CANVAS")

header = st.container()
instructions = st.container()

user_input_zone = st.container()

with header:
    st.title("Table of Contents Scanner")
    st.text("Add very brief intro")

with instructions:
    st.write("Instructions blah blah blah")

with user_input_zone:
    if 'image_captured' not in st.session_state.keys():
        print("Image initialized to none")
        st.session_state['image_captured'] = None

    if 'canvas' not in st.session_state.keys():
        print("Canvas initialized to null")
        st.session_state['canvas'] = None
        st.session_state['contour_gathered'] = False


    getCameraDataThree()

    if (st.session_state['image_captured'] != None):
        print("have already captured img, decide what to show")
        print("State Vars : ")
        print("Captured img :", st.session_state['image_captured'])
        print("Canvas :", st.session_state['canvas'])
        print("Contour Gathered :", st.session_state['contour_gathered'])




        if not st.session_state['contour_gathered']:
            displayCanvasForEditing()
        else:
            displayCanvasResults()


#    st.write_stream    check this out


#get the webcam thingy working
#get two sliders
#get webcam showing up
#add button for webcam
#make button actually take picture
#display the result picture, cropped around the contour
#ask the user if this result looks good
#figure out how to go back to taking a picture step