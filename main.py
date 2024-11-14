import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
#from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
import pandas as pd

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
    # Original unaltered data
    original_objects = st.session_state['canvas']["objects"]

    objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow
    print("SHOWING DATA FOR CONTOUR")
    print(objects)

    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    #TODO, make this display the image with their polygon on it
    st.dataframe(objects)

    #NOTE: If I use Image.crop from Pillow with a rectangular selection, its very easy to crop around a rectangle

    #get the column in objects that contains the coords of the polygon
    poly_path = original_objects[0]['path']
    print(poly_path)
    print(type(poly_path[0]))
    #extract the coordinates from the list of points. In the path they are stuctured like [type, x, y], and the path ends with a ['z']
    polygon_points = np.array([[point[1], point[2]] for point in poly_path if point[0] != 'z'])

    print("Extracted Polygon Points:", polygon_points)


    img = Image.open(st.session_state['image_captured'])
    rgb_img = np.array(img)

    #create a mask of all 0s
    mask = np.zeros((rgb_img.shape[0],rgb_img.shape[1]) , dtype=np.uint8)
    
    #color everything in the polygon as 255
    polygon = np.array([polygon_points], dtype=np.int32)
    cv.fillPoly(mask, polygon, 255)

    #get a copy of the image, but have the pixels outside the polygon all be one color
    polygon_og_color_mask = rgb_img.copy()
    polygon_og_color_mask[mask == 0] = [255, 255, 255]

    mean_val = cv.mean(polygon_og_color_mask[mask])
    print("MEAN")
    print(mean_val)
    print(rgb_img.shape)
    st.image(polygon_og_color_mask)
    
    mean_val = cv.mean(rgb_img, mask=mask)

    
    st.session_state['mask'] = polygon_og_color_mask
    st.session_state['poly'] = polygon
    

header = st.container()
instructions = st.container()
user_input_zone = st.container()
computer_vision_zone = st.container()

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

with computer_vision_zone:
    if st.session_state['contour_gathered']:

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