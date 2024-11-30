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
result_edit_zone = st.container()

def getCameraDataThree():
    camera_result = st.camera_input("Take a level, well lit picture of the table of contents here", key="camera")

    if camera_result:
        
        #THIS IS CAUSING INF LOOP, SINCE MAIN KEEPS RERUNNING AND ALL THE VARS GET RESET
        
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

            st.cache_data.clear() # This is causing an error for me when I take a picture.
            # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data

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
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        if len(objects) == 1 and  st.session_state['cropped'] == False:
            #as soon as one polygon is drawn, we do not want the user to draw more. 
            st.session_state['canvas'] = canvas_result.json_data
            print("SAVED CANVAS DATA, of ONE CONTOUR")
            st.session_state['cropped'] = True

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
        if len(objects) == 1 and st.session_state['angleFound'] == False:

            st.session_state['angle'] = canvas_result.json_data
            st.session_state['angleFound'] = True
            st.rerun()


def calculate_linked_lists(df):
    # Preprocess the DataFrame
    # ------------------------
    # Create a working copy of the DataFrame and add columns for linked list pointers
    rectangles = df.copy()
    rectangles["Next"] = np.nan  # Pointer to the next rectangle in the linked list
    rectangles["Prev"] = np.nan  # Pointer to the previous rectangle in the linked list

    # Add bounding box boundaries as separate columns for clarity
    rectangles["x_min"] = rectangles["Bounding Box X"]
    rectangles["x_max"] = rectangles["Bounding Box X"] + rectangles["Bounding Box Width"]
    rectangles["y_min"] = rectangles["Bounding Box Y"]
    rectangles["y_max"] = rectangles["Bounding Box Y"] + rectangles["Bounding Box Height"]

    # Remove rows with empty text (to ignore noisy bounding boxes)
    rectangles = rectangles[rectangles['OCR_Text'] != ""]

    # Sort rectangles by x_min (left-to-right order)
    rectangles = rectangles.sort_values(by="x_min").reset_index(drop=False)

    # Iterate through rectangles to calculate linked lists
    # ----------------------------------------------------
    for i, current_rect in rectangles.iterrows():
        # Extract bounds for the current rectangle
        current_y_min = current_rect["y_min"]
        current_y_max = current_rect["y_max"]
        current_x_max = current_rect["x_max"]

        # Iterate through rectangles to the right of the current rectangle
        for j, potential_rect in rectangles.iloc[i+1:].iterrows():
            # Extract bounds for the potential rectangle
            potential_y_min = potential_rect["y_min"]
            potential_y_max = potential_rect["y_max"]
            potential_x_min = potential_rect["x_min"]

            # Check if the rectangle is horizontally to the right. With an added threshold
            if potential_x_min >= (current_x_max - 20):
                # Check for vertical overlap
                if (current_y_min <= potential_y_max and current_y_max >= potential_y_min):
                    # Intersection detected! Add to the linked list.
                    rectangles.at[i, "Next"] = j  # Update "Next" pointer for current rectangle
                    rectangles.at[j, "Prev"] = i  # Update "Prev" pointer for the intersected rectangle
                    break  # Stop after finding the first intersection to the right

    return rectangles

def calculateText(df):
    #copy df and add a col
    lists = df.copy()
    lists["LL_Text"] = None

    #create set for LL Heads
    ll_head_indices = set()

    lists["Next"] = lists["Next"].fillna(-1).astype(int)
    lists["Prev"] = lists["Prev"].fillna(-1).astype(int)

    #convert columns of DF from float to int
    if "Next" in lists.columns:
        lists["Next"] = lists["Next"].apply(lambda x: int(x))
    if "Prev" in lists.columns:
        lists["Prev"] = lists["Prev"].apply(lambda x: int(x))

    #iterate through the df.
    for i, current_rect in lists.iterrows():
        
        if current_rect["Prev"] == -1:
            #it's the head of a list
            ll_head_indices.add(i)

            list_text = []
            list_text.append(current_rect["OCR_Text"])

            next_node_i = int(current_rect["Next"]) #convert to python int, instead of numpy int
            while next_node_i != -1:
                
                next_node_i = int(next_node_i) #convert to python int, instead of numpy int

                if not isinstance(next_node_i, int):
                    raise TypeError(
                        f"Expected integer index, but got {type(next_node_i)}: {next_node_i}"
                    )
        
                #use index to retrieve next node row
                next_node = lists.iloc[next_node_i]

                #add text to list
                list_text.append(next_node["OCR_Text"])

                #update next node pointer
                next_node_i = next_node["Next"]
    
            concatenated_text = " ".join(list_text)
            lists.at[i, "LL_Text"] = concatenated_text

    ll_heads_df = lists.loc[list(ll_head_indices)].reset_index(drop=True)
    return ll_heads_df



    

    #add a row to it to be the "root"
    #generate ids for each element
    #


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
        st.session_state['cropped'] = False


    getCameraDataThree()

    if (st.session_state['image_captured'] != None):
        #print("have already captured img, decide what to show")
        #print("State Vars : ")
        #print("Captured img :", st.session_state['image_captured'])
        #print("Canvas :", st.session_state['canvas'])
        #print("Contour Gathered :", st.session_state['cropped'])

        if not st.session_state['cropped']:
            displayCanvasForEditing()
        else:
            displayCanvasResults()

with angle_input_zone:


    if 'angleFound' not in st.session_state.keys():
        #could retrieve the rectangle here?
        st.session_state['angle'] = None
        st.session_state['angleFound'] = False


    if (st.session_state['image_captured'] != None) and (st.session_state['cropped'] == True):
        st.text("READY FOR LINE")

        if not (st.session_state["angleFound"]):
            displayLineInput()
        else:
            st.text("GOT THE ANGLE BUB")

        #st.rerun()
    else:
        st.text("NOT READY FOR LINE")

with computer_vision_zone:
    if ("comp_vision_completed") not in st.session_state.keys():
        st.session_state['comp_vision_completed'] = False
        st.session_state['df'] = None

    if st.session_state.get("angleFound") and not st.session_state['comp_vision_completed']:
        # Ensure the angle has been calculated and the image is ready
        if "rect" in st.session_state.keys():

            original_objects = st.session_state['angle']["objects"]

            img = st.session_state['rect']

            objects = pd.json_normalize(original_objects) # need to convert obj to str because PyArrow

            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            
            #st.dataframe(objects)

            p1 = (objects['x1'].iloc[0], objects['y1'].iloc[0])
            p2 = (objects['x2'].iloc[0], objects['y2'].iloc[0])

            angle_in_radians = 0
            if (p1[0] < p2[0]):
                angle_in_radians = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            else:
                angle_in_radians = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

            angle_in_degrees = np.degrees(angle_in_radians)

            rotated = ndimage.rotate(img, angle_in_degrees, reshape=False)
            new_width = rotated.shape[1] * 2  # Width is the second dimension
            new_height = rotated.shape[0] * 2  # Height is the first dimension
            rotated = cv.resize(rotated, (new_width, new_height))

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
            
            '''
            equ = cv.equalizeHist(grayscale_img)
            #st.image(equ, caption="Histogram Equalized")
            '''

            '''
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(grayscale_img)
            #st.image(cl1, caption="Adaptive hstEQ")
            '''
            normalizedImg = cv.normalize(grayscale_img, None, 0, 255, cv.NORM_MINMAX)
            #st.image(normalizedImg, caption="Normalized")

            '''
            equ2 = cv.equalizeHist(normalizedImg)
            #st.image(equ2, caption="Normalized, then eqHST")

            eq3 = clahe.apply(normalizedImg)
            #st.image(eq3, caption="Normalized, then adaptice hstEQ")
            
            imgf0 = adaptive_threshold(grayscale_img, block_size=11, C=5)
            #st.image(imgf0, caption="Local Adaptive Threshold on gray")
            '''

            imgf = adaptive_threshold(normalizedImg, block_size=15, C=7)
            #st.image(imgf, caption="Local Adaptive Threshold on normalized")

            '''
            imgf2 = adaptive_threshold(eq3, block_size=11, C=5)
            #st.image(imgf2, caption="Local Adaptive Threshold on normalized + Adaptive hst")

            imgf3 = adaptive_threshold(cl1, block_size=11, C=5)
            #st.image(imgf3, caption="Threshold on adaptive HstEQ")
            '''
            #Rotate manually
            #DONE

            #Binarization
            #DONE - use local adapative T on (Normalized) OR on (Grayscale)

            #Remove noise
            no_noise = noise_removal(imgf)
            st.image(no_noise, caption="Noise Reduction on normalized")

            #Algorithmic deskewing - Skip for now
            
            copy = no_noise.copy()

            #erode in the horizontal especially, to connect the letters
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,3))
            eroded =  cv.erode(no_noise, kernel, iterations=2)

            #find contours NOTE: Look into different hierarchies RETR_TREE, List, ect
            contours, hierarchy = cv.findContours(eroded, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            #remove largest contour
            sorted_contours = sorted(contours, key=lambda c: cv.boundingRect(c)[2] * cv.boundingRect(c)[3], reverse=True)
            cnts = sorted_contours[1:]
            cnts = sorted(cnts, key=lambda c: cv.boundingRect(c)[1])

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
            st.image(eroded, caption = "Using erosion and contour detection for text blocks")
            st.image(copy)

            df = pd.DataFrame(data)
            st.dataframe(df)

            #group text horizontally, create LL 
            rectangle_lists = calculate_linked_lists(df)

            st.dataframe(rectangle_lists)

            linked_list_df = calculateText(rectangle_lists)

            st.session_state['comp_vision_completed'] = True
            st.session_state['df'] = linked_list_df

            #concatenate the text, based off the linked list
            

            
            st.dataframe(linked_list_df)
            
            # Make dataframe editable
            
            #st.dataframe(edited_df)

            #st.button(label="Generate Hierarchy", on_click=generate_hierarchy, args=(edited_df,))

            #generate_hierarchy(edited_df)


#    st.write_stream    check this out

with result_edit_zone:
    if (st.session_state["comp_vision_completed"]):
        df = st.session_state["df"]

        df['Author'] = ''
        df['Title'] = ''

        edited_df = st.data_editor(df, column_order=['LL_Text', 'Author', 'Title'], num_rows='dynamic', disabled=False)

        sorted_df = edited_df.sort_values(by="y_min").reset_index(drop=True)


        #st.dataframe(sorted_df) 
        #print(sorted_df.columns.tolist())

        #add a row for the root
        root_row = pd.DataFrame({
            'index': [None],
            'Bounding Box X': [None],
            'Bounding Box Y': [None],
            'Bounding Box Width': [None],
            'Bounding Box Height': [None],
            'OCR_Text': ['ROOT'],
            'Next': [None],
            'Prev': [None],
            'x_min': [None],
            'x_max': [None],
            'y_min': [None],
            'y_max': [None],
            'LL_Text': [None],
            'Author': [None],
            'Title': [None]
        })

        df = pd.concat([root_row, sorted_df], ignore_index=True)

        #add a column for "type". make '' for all except root.
        df['Type'] = ''
        df.at[0, 'Type'] = "ROOT"
        df.at[0, 'LL_Text'] = "ROOT"

        #add a path column. Make sure it's dtype=object 
        df['Path'] = [[] for _ in range(len(df))]
        #for all except ROOT the contents of this col will be like ['ROOT', ' ROW TEXT']
        df.at[0, 'Path'] = ['ROOT']
        for i in range(1, len(df)):
            df.at[i, 'Path'] = ['ROOT', df.at[i, 'LL_Text']]

        st.dataframe(df)

        df_subset = df[["LL_Text","Type","Path"]]
        df_subset.rename(columns={"LL_Text": "Text"})
        json = df_subset.to_json(orient='records')

        print(json)
        
        #add a 
#get the webcam thingy working
#get two sliders
#get webcam showing up
#add button for webcam
#make button actually take picture
#display the result picture, cropped around the contour
#ask the user if this result looks good
#figure out how to go back to taking a picture step