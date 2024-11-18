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
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    blur = cv.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    for c in contours:
        rect = cv.boundingRect(c)
        x,y,w,h = rect
        cv.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv.minAreaRect(largestContour)
    cv.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

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

            st.cache_data.clear() # This is causing an error for me when I take a picture, not sure why.

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

def calculateAngle():
    original_objects = st.session_state['angle']["objects"]

    img = st.session_state['rect']

    objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow
    #print("SHOWING DATA FOR CONTOUR")
    #print(objects)

    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    #TODO, make this display the image with their polygon on it
    st.dataframe(objects)

    p1 = (objects['x1'].iloc[0], objects['y1'].iloc[0])
    p2 = (objects['x2'].iloc[0], objects['y2'].iloc[0])

    #NOTE: 
    print(p1)
    print(p2)

    angle_in_radians = 0
    if (p1[0] < p2[0]):
        angle_in_radians = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    else:
        angle_in_radians = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

    angle_in_degrees = np.degrees(angle_in_radians)

    #adjusted_angle = -angle_in_degrees

    #may need to use the midpoint of the IMAGE instead or the line points. Since line point coords are defined by their center pt
    #rotation_matrix = cv.getRotationMatrix2D((int(midpoint[0]), int(midpoint[1])), angle_in_degrees, 1)
    #rotated_img = cv.transform(st.session_state['rect'], rotation_matrix)

    rotated = ndimage.rotate(img, angle_in_degrees, reshape=False)

    st.image(rotated)

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
            calculateAngle()

        #st.rerun()
    else:
        st.text("NOT READY FOR LINE")


with computer_vision_zone:
    if st.session_state['contour_gathered'] and False:


        mask = st.session_state['mask']
        polygon = st.session_state['poly'] #(x,y)
        x,y,w,h  = cv.boundingRect(polygon)

        gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        cropped_image = mask[y:y+h, x:x+w]

        st.image(cropped_image)

        #note, maybe should be rgb -> gray
        grayscale_cropped = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

        larger_cropped = cv.resize(grayscale_cropped, (grayscale_cropped.shape[1] * 2, grayscale_cropped.shape[0] * 2))
        st.image(larger_cropped)

        deskewed = deskew(larger_cropped)

        st.image(deskewed)




#    st.write_stream    check this out


#get the webcam thingy working
#get two sliders
#get webcam showing up
#add button for webcam
#make button actually take picture
#display the result picture, cropped around the contour
#ask the user if this result looks good
#figure out how to go back to taking a picture step