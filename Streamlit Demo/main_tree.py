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

'''
tabs = st.tabs(["Grid", "Underlying Data", "Grid Options", "Grid Return"])
drag_container = st.container()

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
'''
face_detection()
# tabs =  st.tabs(['Selected Rows','gridoptions','grid_response'])

# with tabs[0]:


onRowDragEnd = JsCode("""
function onRowDragEnd(e) {
    console.log('onRowDragEnd', e);
                      
    var overNode = e.overNode;
    console.log("OVERNODE", overNode);
    if (!overNode) {
        return;
    }
                      
    var node = e.node;
    console.log("NODE", node);
                      
    var movingData = node.data;
                      
    // take new parent path from parent, if data is missing, means it's the root node, which has no data.
    var newParentPath = overNode.data
        ? overNode.data.path
        : [];


    console.log("New Parent Path :", newParentPath);
    console.log("Node Path: ", movingData.path);
    var equal = true;
    if (newParentPath.length !== movingData.path.length) {
        equal = false;  
    } else {
        newParentPath.forEach(function (item, index) {
            console.log("Comparing :");
            console.log(item, movingData.path[index]);
                      
            if (movingData.path[index] !== item) {
                equal = false;
            }
        });
    }

    //is false if dropped on the same cell as picked up from
    var needToChangeParent = !equal;

                      
    console.log("Change parent? ", needToChangeParent);
    

    //check to make sure we are not moving a parent into one of its children
    var invalidMode = false;
    let children = [...(node.childrenAfterGroup || [])];
    while (children.length) {
        const child = children.shift();
        if (!child) {
            continue;
        }

        if (child.key === overNode.key) {
            invalidMode = true;
        }

        //add "granchildren" to the queue
        if (child.childrenAfterGroup && child.childrenAfterGroup.length) {
            children.push(...child.childrenAfterGroup);
        }
    }
                      
    if (invalidMode) {
        console.log("invalid move");
    }
    
    console.log("Invalid move? ", invalidMode);

    function moveToPath(newParentPath, node, allUpdatedNodes) {
                      
        console.log("IN FUNCTION");
        // last part of the file path is the file name
        var oldPath = node.data.path;
        var fileName = oldPath[oldPath.length - 1];
        var newChildPath = newParentPath.slice();
        newChildPath.push(fileName);

        node.data.path = newChildPath;

        allUpdatedNodes.push(node.data);

        if (node.childrenAfterGroup) {
            node.childrenAfterGroup.forEach((childNode) => {
            moveToPath(newChildPath, childNode, allUpdatedNodes);
            });
        }
    }
      
    if (needToChangeParent && !invalidMode) {
        var updatedRows = [];
        moveToPath(newParentPath, e.node, updatedRows);
                      
        //refreshClientSideRowModel();
        
                      
        var element = document.getElementById("gridContainer");
        console.log("DIV", element);
        
        //element.applyTransaction({
        //    update: updatedRows,
        //});
        //element.clearFocusedCell();
        //DOES NOT WORK, window.gridApi is undefined due to hard-coded AG-Grid onGridReady
        // window.gridApi.applyTransaction({
        //    update: updatedRows,
        //});

        //window.gridApi.clearFocusedCell();
    }
}
""")

getRowNodeId = JsCode("""
function getRowNodeId(data) {
    return data.id
}
""")

onGridReady = JsCode("""
function onGridReady(params) {
    console.log("Grid is ready");
    window.gridApi = params.api; // Assign the grid API to a global variable
    console.log("Grid API set:", window.gridApi);
}
""")

onRowDragMove = JsCode("""
function onRowDragMove(event) {
    //console.log("RowDragMove", event);
    //var movingNode = event.node;
    //var overNode = event.overNode;

    //console.log("Moving :", movingNode);
    //console.log("Over :", overNode);
              
}
""")
getDataPath = JsCode("""
function getDataPath(data) {
    return data.path;
} 
""")

getTitle = JsCode("""
function getDataTitle(params) {
    //console.log("PARAMS", params);
    //console.log(params.data.title);
                  
    //have the grouped column be by the title
    return params.data.title;
}
""")

def createGrid(db):
    print("IN BUILDER")
    gb = GridOptionsBuilder.from_dataframe(db)

    gb.configure_default_column(rowDrag = False, rowDragManaged = False, rowDragEntireRow = False, rowDragMultiRow=False)
    gb.configure_grid_options(rowDragManaged = False, onRowDragEnd = onRowDragEnd, getRowNodeId = getRowNodeId, groupDefaultExpanded=-1, treeData=True, getDataPath=getDataPath, 
    autoGroupColumnDef=dict(
        rowDrag=True,
        minWidth=300, 
        pinned="The Groups", 
        cellRendererParams=dict(suppressCount=True, innerRenderer=getTitle)
    ))
    gridOptions = gb.build()

    grid = AgGrid(db,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        update_on=['rowDragEnd'],)
    
    return grid
#https://discuss.streamlit.io/t/drag-and-drop-rows-in-a-dataframe/33077
with st.container():

    if 'db' not in st.session_state.keys():
        data = pd.read_json('contents_tree.json')
        st.session_state['db'] = data
        print("INITIALIZED")
    db = st.session_state['db']
    resp = createGrid(db)
        
    if (resp.event_data):

        #check if the paths are any different
        before_df = st.session_state['db']
        current_df = resp['data']

        before_paths = set(tuple(path) for path in before_df["path"])
        current_paths = set(tuple(path) for path in current_df["path"])
        
        difference = before_paths != current_paths

        if (difference):
            print("NEW DATA")
            st.write(resp.event_data)
            st.session_state['db'] = resp['data']
            st.rerun()
        else:
            print("IDENTICAL")




#st.write(grid['data'])
#st.write(grid.grid_options)

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