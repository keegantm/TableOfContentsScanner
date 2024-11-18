import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
#from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from scipy import ndimage

'''
Keegan's Todos:
- Change polygon to rect
- See how hard getting the angle would be 

'''

#https://www.youtube.com/watch?v=ADV-AjAXHdc&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i&index=4
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return (image)


def adaptive_threshold(image, block_size=11, C=2):
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)

# Opening
def morphological_opening(image, kernel_size=(3, 3)):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

header = st.container()
instructions = st.container()
crop_input_zone = st.container()
angle_input_zone = st.container()
computer_vision_zone = st.container()

def getCameraDataThree():
    camera_result = st.camera_input("Take a level, well lit picture of the table of contents here", key="camera")

    if camera_result:
        
        #THIS IS CAUSING INF LOOP, SINCE MAIN KEEPS RERUNNING AND ALL THE VARS GET RESET
        
        if (st.session_state['image_captured'] != camera_result):
            print("NEW PHOTO RESETTING STATES")
            st.session_state['canvas'] = None
            st.session_state['contour_gathered'] = False

            if ('angleFound' in st.session_state.keys()):
                st.session_state['angleFound'] = False
            
            if ('angle' in st.session_state.keys()):
                st.session_state['angle'] = None

            if ('rect' in st.session_state.keys()):
                st.session_state['rect'] = None

            st.cache_data.clear()

        st.session_state['image_captured'] = camera_result

def displayCanvasForEditing():

    img = Image.open(st.session_state['image_captured'])
    rgb_img = np.array(img)

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

    if canvas_result.json_data is not None:
        #print(canvas_result.json_data)
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        if len(objects) == 1 and  st.session_state['contour_gathered'] == False:
            #as soon as one polygon is drawn, we do not want the user to draw more. 
            st.session_state['canvas'] = canvas_result.json_data
            print("SAVED CANVAS DATA, of ONE CONTOUR")
            st.session_state['contour_gathered'] = True

            original_objects = canvas_result.json_data["objects"]
            x = original_objects[0]['left']
            y = original_objects[0]['top']
            w = original_objects[0]['width']
            h = original_objects[0]['height']

            img = Image.open(st.session_state['image_captured'])
            rgb_img = np.array(img)

            cropped_image = rgb_img[y:y+h, x:x+w]

            st.session_state['rect'] = cropped_image
            st.rerun()


def displayCanvasResults():
    original_objects = st.session_state['canvas']["objects"]

    objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow

    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    st.dataframe(objects)
    
    st.image(st.session_state['rect'])


def displayLineInput():
    if 'rect' not in st.session_state.keys():
        print("WHATNJKSBDKJNASLKDJKLAS")
        return
    
    st.write("Click and draw a line to rotate the image, from LEFT to RIGHT")

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

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        print(len(objects))
        if len(objects) == 1 and st.session_state['angleFound'] == False:
            print("Found the angle")

            st.session_state['angle'] = canvas_result.json_data
            st.session_state['angleFound'] = True
            print(st.session_state['angleFound'])
            st.rerun()


with header:
    st.title("Table of Contents Scanner")
    st.text("Add very brief intro")

with instructions:
    st.write("Instructions blah blah blah")

with crop_input_zone:
    if 'image_captured' not in st.session_state.keys():
        #print("Image initialized to none")
        st.session_state['image_captured'] = None

    if 'canvas' not in st.session_state.keys():
        #print("Canvas initialized to null")
        st.session_state['canvas'] = None
        st.session_state['contour_gathered'] = False


    getCameraDataThree()

    if (st.session_state['image_captured'] != None):
        #print("have already captured img, decide what to show")
        #print("State Vars : ")
        #print("Captured img :", st.session_state['image_captured'])
        #print("Canvas :", st.session_state['canvas'])
        #print("Contour Gathered :", st.session_state['contour_gathered'])

        if not st.session_state['contour_gathered']:
            displayCanvasForEditing()
        else:
            displayCanvasResults()

with angle_input_zone:

    if 'angleFound' not in st.session_state.keys():
        #could retrieve the rectangle here?
        st.session_state['angle'] = None
        st.session_state['angleFound'] = False


    if (st.session_state['image_captured'] != None) and (st.session_state['contour_gathered'] == True):
        st.text("READY FOR LINE")

        if not (st.session_state["angleFound"]):
            displayLineInput()
        else:
            st.text("GOT THE ANGLE BUB")

        #st.rerun()
    else:
        st.text("NOT READY FOR LINE")

with computer_vision_zone:
    if st.session_state.get("angleFound"):
        # Ensure the angle has been calculated and the image is ready
        if "rect" in st.session_state:
            original_objects = st.session_state['angle']["objects"]

            img = st.session_state['rect']

            objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow

            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            
            #st.dataframe(objects)

            p1 = (objects['x1'].iloc[0], objects['y1'].iloc[0])
            p2 = (objects['x2'].iloc[0], objects['y2'].iloc[0])

            print(p1)
            print(p2)

            angle_in_radians = 0
            if (p1[0] < p2[0]):
                angle_in_radians = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            else:
                angle_in_radians = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

            angle_in_degrees = np.degrees(angle_in_radians)

            rotated = ndimage.rotate(img, angle_in_degrees, reshape=False)

            rotated = cv.resize(rotated, (rotated.shape[0] * 2, rotated.shape[1] * 2))

            # Convert to grayscale
            grayscale_img = cv.cvtColor(rotated, cv.COLOR_RGB2GRAY)

            # Display the rotated image
            st.text("Rotated Image:")
            st.image(rotated, caption="Rotated img")

            # Display the grayscale image
            st.text("Grayscale Image:")
            st.image(grayscale_img, caption="Grayscale Image for OCR")

            # Store the grayscale image in session state for OCR
            st.session_state["grayscale_img"] = grayscale_img

            equ = cv.equalizeHist(grayscale_img)
            st.image(equ, caption="Histogram Equalized")

            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(grayscale_img)
            st.image(cl1, caption="Adaptive hstEQ")

            normalizedImg = cv.normalize(grayscale_img, None, 0, 255, cv.NORM_MINMAX)
            st.image(normalizedImg, caption="Normalized")

            equ2 = cv.equalizeHist(normalizedImg)
            st.image(equ2, caption="Normalized, then eqHST")

            eq3 = clahe.apply(normalizedImg)
            st.image(eq3, caption="Normalized, then adaptice hstEQ")
            
            imgf0 = adaptive_threshold(grayscale_img, block_size=11, C=5)
            st.image(imgf0, caption="Local Adaptive Threshold on gray")

            imgf = adaptive_threshold(normalizedImg, block_size=15, C=7)
            st.image(imgf, caption="Local Adaptive Threshold on normalized")

            imgf2 = adaptive_threshold(eq3, block_size=11, C=5)
            st.image(imgf2, caption="Local Adaptive Threshold on normalized + Adaptive hst")

            imgf3 = adaptive_threshold(cl1, block_size=11, C=5)
            st.image(imgf3, caption="Threshold on adaptive HstEQ")

            #Rotate manually
            #DONE

            #Binarization
            #DONE - use local adapative T on (Normalized) OR on (Grayscale)

            #Remove noise
            no_noise = noise_removal(imgf)
            st.image(no_noise, caption="Noise Reduction on normalized")

            #Algorithmic deskewing - Skip for now

            #Contour stuff. Get rid of borders, isolate chunks of txt, ect

#    st.write_stream    check this out


#get the webcam thingy working
#get two sliders
#get webcam showing up
#add button for webcam
#make button actually take picture
#display the result picture, cropped around the contour
#ask the user if this result looks good
#figure out how to go back to taking a picture step