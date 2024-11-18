import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import datetime
import numpy as np
import requests
import uuid
import json
import cv2 as cv

def face_detection():

    img_file_buffer = st.camera_input("Webcam")

    print(img_file_buffer)

    if img_file_buffer is not None:

        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)

        face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

        #ret, frame = vid.read()
        gray_image = cv.cvtColor(cv2_img, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            cv.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
        st.image(cv2_img)



#https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/9a8503b806ef3237f2b6c21cf2ad4dc97dec568c/20_cell_renderer_class_example.py 
#good for buttons within the DF

#https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/9a8503b806ef3237f2b6c21cf2ad4dc97dec568c/82_Handling_Grid_events.py
#displays how to use event handlers. Could use to change hierarchy

#https://www.ag-grid.com/javascript-data-grid/tree-data/
#Tree-like hierarchy from a json

#https://www.ag-grid.com/javascript-data-grid/row-dragging/
#Talks about dragging rows, with example of tree-strucutre 

#https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input
#Has info for taking pictures, and using within open cv

#https://thiagoalves.ai/building-webcam-streaming-applications-with-streamlit-and-opencv/
#info on using webcam

#https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment

# Define the Streamlit containers
header = st.container()

with header:
    st.title("StreamLit Demo")
    st.text("My test application, to determine if Streamlit is a good fit for my project")

#read in the JSON. Our CV algo would process a video/image, try to turn it into JSON in a certain format, and would be used here
with open('contents_tree.json', 'r') as file:
    data = json.load(file)
    #df = pd.read_json(file)
    df = pd.DataFrame(data)
    print(df)
file.close()

gridOptions = {
    "treeData": True,
    "getDataPath": JsCode("function(data) { return data.path; }"),
    "groupDefaultExpanded" : True,
    "rowSelection": "single",
    "rowDragManaged": True,
    "rowDragEntireRow": True,
    "autoGroupColumnDef": {
        "cellRendererParams": {
            "suppressCount": True
        }
    },
    "columnDefs": [
        {"field": "chapter", "headerName": "Chapter/Sub-Chapter"},
        {"field": "author"},
        {"field": "pageCount"}
    ],
    "rowData": data,
}


tabs = st.tabs(["Grid", "Underlying Data", "Grid Options", "Grid Return"])

with tabs[0]:
    r = AgGrid(
        None,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        # update_mode=GridUpdateMode.SELECTION_CHANGED,
        key="an_unique_key",
        
    )

with tabs[1]:
    st.write(data)

with tabs[2]:
    st.write(gridOptions)

face_detection()
# tabs =  st.tabs(['Selected Rows','gridoptions','grid_response'])

# with tabs[0]:

'''
Project Steps:
INPUT
- Get webcam video working
    - Optional: Add bounding corner for the top L corner of the book's page. Would help us find the correct contour to examine
-Show the img. Allow the user to retake pic, or possibly define the boundary of the contour here
-Alternative strategy: use shape detection, the big rectangle should be the book. This is less idiot proof though

CV STUFF
- apply preprocessing
- draw bounding boxes around the contents 
    - Could use to apply hierarchy
- read text of each box

POSSIBLE NLP STEP
- Could try to clean up spelling mistakes from OCR
- If we can't get a hierarchy based off of the positions of the bounding boxes, we can try to use word embeddings to determine hierarchy

JSONIFY CV RESULTS
- Write json of the predicted hierarchy and words

CREATE TREE GRID
- Need to write additional functionalitites into the AG grid   
    Primary Objective:
        -Allow drag and drop between hierarchies, of a single row at a time
    Secondary Objectives:
        - Delete Row
        - Add row
        - Merge row A into Row B (Use case is if there's a two line title that was interpretted as two separate rows) 

-DISPLAY AND EDITTING PHASE
- User views the tree grid, corrects spelling, words, hierarchy. This is super easy with ag_grid

OPTIONAL:
- Export the JSON of the tree grid
'''