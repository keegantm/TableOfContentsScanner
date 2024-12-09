import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
#from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from scipy import ndimage
import pytesseract
from pytesseract import Output
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from utils import *

#streamlit content containers
header = st.container()
instructions = st.container()
image_input_and_crop_zone = st.container()
angle_input_zone = st.container()
computer_vision_zone = st.container()
result_edit_zone = st.container()
hierarchy_edit_zone = st.container()

#function to get the user's camera input
def getCameraData():
    camera_result = st.camera_input("Take a level, well lit picture of the table of contents here", key="camera")

    if camera_result:

        #since we have a new image, reset state variables to re-run the program with this new img        
        if (st.session_state['image_captured'] != camera_result):
            print("NEW PHOTO RESETTING STATES")
            st.session_state['canvas'] = None
            st.session_state['cropped'] = False

            if ('angleFound' in st.session_state.keys()):
                st.session_state['angleFound'] = False
            
            if ('angle' in st.session_state.keys()):
                st.session_state['angle'] = None

            if ('rect' in st.session_state.keys()):
                st.session_state['rect'] = None

            if ('comp_vision_completed' in st.session_state.keys()):
                st.session_state['comp_vision_completed'] = False
                
            if ('df' in st.session_state.keys()):
                st.session_state['df'] = None

            if ('hierarchy_prepared' in st.session_state.keys()):
                st.session_state['hierarchy_prepared'] = False

            if ('hierarchy_df' in st.session_state.keys()):
                st.session_state['hierarchy_df'] = None

            st.cache_data.clear() # This is causing an error for me when I take a picture.
            # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data

        #store the image taken in state var
        st.session_state['image_captured'] = camera_result

#function for canvas editting
def cropImage():

    img = Image.open(st.session_state['image_captured'])
    rgb_img = np.array(img)

    #display image on the screen, allow user to crop
    canvas_result = st_canvas(
        background_image=Image.open(st.session_state['image_captured']),
        drawing_mode='rect',
        stroke_color='red',
        height = rgb_img.shape[0],
        width = rgb_img.shape[1],
        fill_color="",
        stroke_width=5,
        display_toolbar=False,
        update_streamlit=True,
    )

    #canvas_result.json_data contains drawn rectangle information
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        if len(objects) == 1 and  st.session_state['cropped'] == False:
            #as soon as one polygon is drawn, we do not want the user to draw more. 
            st.session_state['canvas'] = canvas_result.json_data
            #print("SAVED CANVAS DATA, of ONE CONTOUR")
            st.session_state['cropped'] = True

            #get rectangle dimensions
            original_objects = canvas_result.json_data["objects"]
            x = original_objects[0]['left']
            y = original_objects[0]['top']
            w = original_objects[0]['width']
            h = original_objects[0]['height']

            #use recctangle dimensions to crop the image
            img = Image.open(st.session_state['image_captured'])
            rgb_img = np.array(img)
            cropped_image = rgb_img[y:y+h, x:x+w]

            st.session_state['rect'] = cropped_image
            st.rerun()

#function for displaying the results of a crop
def displayCropResults():
    original_objects = st.session_state['canvas']["objects"]

    objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow

    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    #st.dataframe(objects)
    
    st.image(st.session_state['rect'])

#function allowing the user to draw a line, in order to define a NEW X axis on an image & rotate it.
def displayLineInput():

    #debug stuff
    if 'rect' not in st.session_state.keys():
        print("WHATNJKSBDKJNASLKDJKLAS")
        return
    
    #instructions!
    st.write("Underline a line of text, in order to rotate the image to be level.")

    #get the cropped image
    image = st.session_state['rect'].astype('uint8')

    #create a canvas for the line input 
    canvas_result = st_canvas(
        background_image = Image.fromarray(image, 'RGB'),
        drawing_mode='line',
        stroke_color='red',
        height = image.shape[0],
        width = image.shape[1],
        fill_color="",
        stroke_width=5,
        display_toolbar=False,
        update_streamlit=True,
    )

    #get the drawn line
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        if len(objects) == 1 and st.session_state['angleFound'] == False:

            st.session_state['angle'] = canvas_result.json_data
            st.session_state['angleFound'] = True
            st.rerun()

with header:
    st.title("Table of Contents Scanner")
    st.text("An interactive app for table of contents extraction, and editting")

with instructions:
    st.write("Instructions blah blah blah")

with image_input_and_crop_zone:
    #initialize state vars 
    if 'image_captured' not in st.session_state.keys():
        #print("Image initialized to none")
        st.session_state['image_captured'] = None
    if 'canvas' not in st.session_state.keys():
        #print("Canvas initialized to null")
        st.session_state['canvas'] = None
        st.session_state['cropped'] = False

    #have the user input an image
    getCameraData()

    #allow the user to crop an image, if an image has been taken
    if (st.session_state['image_captured'] != None):
        if not st.session_state['cropped']:
            cropImage()
        else:
            displayCropResults()

with angle_input_zone:
    #initialize state vars
    if 'angleFound' not in st.session_state.keys():
        #could retrieve the rectangle here?
        st.session_state['angle'] = None
        st.session_state['angleFound'] = False
    
    #if a cropped image exists, then allow a line to be drawn
    if (st.session_state['image_captured'] != None) and (st.session_state['cropped'] == True):

        if not (st.session_state["angleFound"]):
            displayLineInput()

        #st.rerun()

#this zone is dedicated to preprocessing the image, applying OCR to it, and joining horizontal OCR results
with computer_vision_zone:
    #initialize variables
    if ("comp_vision_completed") not in st.session_state.keys():
        st.session_state['comp_vision_completed'] = False
        st.session_state['df'] = None

    if st.session_state.get("angleFound") and not st.session_state['comp_vision_completed']:
        # Ensure the angle has been calculated and the image is ready
        if "rect" in st.session_state.keys():

            #---------
            # Preprocess image
            #---------

            #get the angle for rotating img
            original_objects = st.session_state['angle']["objects"]
            img = st.session_state['rect']
            objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow

            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            
            p1 = (objects['x1'].iloc[0], objects['y1'].iloc[0])
            p2 = (objects['x2'].iloc[0], objects['y2'].iloc[0])

            #using line points, get img rotation angle
            angle_in_radians = 0
            if (p1[0] < p2[0]):
                angle_in_radians = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            else:
                angle_in_radians = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

            angle_in_degrees = np.degrees(angle_in_radians)

            #rotate and resize img
            rotated = ndimage.rotate(img, angle_in_degrees, reshape=False)
            new_width = rotated.shape[1] * 2  # Width is the second dimension
            new_height = rotated.shape[0] * 2  # Height is the first dimension
            rotated = cv.resize(rotated, (new_width, new_height))

            #convert to grayscale
            grayscale_img = cv.cvtColor(rotated, cv.COLOR_RGB2GRAY)

            # Display the rotated image
            st.text("Rotated Image:")
            st.image(rotated, caption="Rotated and Resized Image")

            # Display the grayscale image
            #st.text("Grayscale Image:")
            #st.image(grayscale_img, caption="Grayscale Image")

            #store the grayscale image in session state for OCR
            st.session_state["grayscale_img"] = grayscale_img
            
            #normalize the img
            normalizedImg = cv.normalize(grayscale_img, None, 0, 255, cv.NORM_MINMAX)
            #st.image(normalizedImg, caption="Normalized")

            binary = adaptive_threshold(normalizedImg, block_size=15, C=7)
            #st.image(binary, caption="Local Adaptive Threshold on normalized")
    
            #Remove noise
            no_noise = noise_removal(binary)
            st.image(no_noise, caption="Noise-Reduced Binary Image")

            #Algorithmic deskewing - Skip for now
            
            copy = no_noise.copy()

            #--------------
            # This next section is decicated to extracting the position of text elements
            #--------------
            #erode in the horizontal especially, to connect the letters
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,3))
            eroded =  cv.erode(no_noise, kernel, iterations=2)

            #find contours NOTE: Look into different hierarchies RETR_TREE, List, ect
            contours, hierarchy = cv.findContours(eroded, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            #remove largest contour
            sorted_contours = sorted(contours, key=lambda c: cv.boundingRect(c)[2] * cv.boundingRect(c)[3], reverse=True)
            cnts = sorted_contours[1:]
            cnts = sorted(cnts, key=lambda c: cv.boundingRect(c)[1])

            #------------
            # Apply OCR within each individual text bounding box
            #------------
            data = []
            for c in cnts:
                x, y, w, h = cv.boundingRect(c)
                if (w * h) > 300: #only include big enough 
                    #region of interest, in the rectangle
                    roi = no_noise[y:y+h, x:x+w]

                    #add a rectangle to the img
                    cv.rectangle(eroded, (x, y), (x+w, y+h), (100, 100, 100), 2)
                    cv.rectangle(copy, (x, y), (x+w, y+h), (100, 100, 100), 2)

                    #use ocr on the ROI
                    ocr_result = pytesseract.image_to_string(roi)

                    data.append({
                        'Bounding Box X': x,
                        'Bounding Box Y': y,
                        'Bounding Box Width': w,
                        'Bounding Box Height': h,
                        'OCR_Text': ocr_result
                    })
            st.image(eroded, caption = "Text Blocks Found Using Erosion and Countour Detection")
            st.image(copy, caption = "Text Blocks Found Overlayed on the Preproccessed Image")

            #Data frame of OCR results, and text block info
            df = pd.DataFrame(data)
            #st.dataframe(df)

            #Link each text block to any existing text block to its right
            rectangle_lists = calculate_linked_lists(df)

            #st.dataframe(rectangle_lists)

            #concatenate text according to the linked lists
            linked_list_df = calculateText(rectangle_lists)

            st.session_state['comp_vision_completed'] = True
            st.session_state['df'] = linked_list_df

def handle_data_editor_change():
    #here, we can inform the hierarchy that new data was input
    st.session_state['hierarchy_prepared'] = False

with result_edit_zone:
    #initialize variables
    if ("hierarchy_prepared") not in st.session_state.keys():
        st.session_state['hierarchy_prepared'] = False
        st.session_state['hierarchy_df'] = None
    

    if (st.session_state["comp_vision_completed"]):

        # ------------------------
        # Prepare df for content editting
        # ------------------------

        #get linked list dataframe
        df = st.session_state["df"]
        #rename LL_Text to text
        df.rename(columns={"LL_Text": "Text"}, inplace=True)

        #Add columns for the user to designate some text as Author/Title
        if 'Author' not in df.columns:
            df['Author'] = False
        if 'Title' not in df.columns:
            df['Title'] = False

        st.write("# Edit Result Content!")
        #create data editor and return result as a dataframe
        edited_df = st.data_editor(
            df, 
            column_order=['Text', 'Author', 'Title'], 
            num_rows='dynamic', 
            disabled=False,
            on_change=handle_data_editor_change)   

        #create hierarchy ready data from editor, if needed
        if not st.session_state['hierarchy_prepared']:

            hierarchical_df = prepare_data_for_hierarchy(edited_df)

            st.session_state['hierarchy_prepared'] = True
            st.session_state['hierarchy_df'] = hierarchical_df
        #else:
            #print("CANNOT INITIALIZE DATA FOR HIERARCHY")
        
with hierarchy_edit_zone:

    #only display a hierarchy editor if we have data to display
    if (st.session_state['hierarchy_prepared']):
        
        #get data for hierarchy
        df = st.session_state['hierarchy_df']

        st.write("# Edit Result Hierarchy!")
        
        #create interactive Ag-Grid
        resp = createGrid(df)
        
        #event data is current data of grid
        if (resp.event_data):

            #check if the paths are any different within the data
            before_df = st.session_state['hierarchy_df']
            current_df = resp['data']

            before_paths = set(tuple(path) for path in before_df["Path"])
            current_paths = set(tuple(path) for path in current_df["Path"])
            
            difference = before_paths != current_paths

            #if the paths are different, then update data used to create grid, and re-run
            if (difference):
                #print("NEW DATA")
                #st.write(resp.event_data)
                st.session_state['hierarchy_df'] = resp['data']
                st.rerun()
            #else:
                #print("IDENTICAL")


#Todos:
# 1. Clean up code
# 2. Fix instructions
# 3. See if we can do a better crop + image tweak