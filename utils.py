import cv2 as cv
import numpy as np 
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

'''
-------------------
Functions for computer vision
-------------------
'''

#Function for reducing noise in an image. Followed the tutorial below
#https://www.youtube.com/watch?v=ADV-AjAXHdc&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i&index=4
def noise_removal(image):

    #dilate and erode a tiny amount. Removes very tiny specs of noise
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)

    #use morphology and blur to further reduce noise 
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)

    return (image)

def adaptive_threshold(image, block_size=11, C=2):
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)

'''
---------------
Functions for processing compouter vision results
---------------
'''


'''
Using text block rectangles in a dataframe, link each rectangle to
the closest text block to it's right (if one exists)
'''
def calculate_linked_lists(df):

    rectangles = df.copy()
    rectangles["Next"] = np.nan  #pointer to the next rectangle in the linked list
    rectangles["Prev"] = np.nan  #pointer to the previous rectangle in the linked list

    #redundant, but add these columns for my sanity
    rectangles["x_min"] = rectangles["Bounding Box X"]
    rectangles["x_max"] = rectangles["Bounding Box X"] + rectangles["Bounding Box Width"]
    rectangles["y_min"] = rectangles["Bounding Box Y"]
    rectangles["y_max"] = rectangles["Bounding Box Y"] + rectangles["Bounding Box Height"]

    #remove rows with empty text (to ignore noisy bounding boxes)
    rectangles = rectangles[rectangles['OCR_Text'] != ""]

    #sort rectangles by x_min (left-to-right order)
    rectangles = rectangles.sort_values(by="x_min").reset_index(drop=False)

    #iterate through rectangles to calculate linked lists
    for i, current_rect in rectangles.iterrows():
        #extract bounds for the current rectangle
        current_y_min = current_rect["y_min"]
        current_y_max = current_rect["y_max"]
        current_x_max = current_rect["x_max"]

        #iterate through rectangles to the right of the current rectangle
        for j, potential_rect in rectangles.iloc[i+1:].iterrows():
            #extract bounds for the potential "Next" rectangle
            potential_y_min = potential_rect["y_min"]
            potential_y_max = potential_rect["y_max"]
            potential_x_min = potential_rect["x_min"]

            #check if the rectangle is to the right. With an added threshold
            if potential_x_min >= (current_x_max - 20):
                
                #make sure the potential rect is aligned with the Y of the current rect
                if (current_y_min <= potential_y_max and current_y_max >= potential_y_min):
                    #Intersection detected! Add to the linked list.
                    rectangles.at[i, "Next"] = j  #update "Next" pointer for current rectangle
                    rectangles.at[j, "Prev"] = i  #update "Prev" pointer for the intersected rectangle
                    break #stop after finding the first intersection to the right

    return rectangles

'''
Using a linked list of rectangles, concatenate the text along the linked list.
Return only the heads of the linked lists
'''
def calculateText(df):
    #copy df and add a col
    lists = df.copy()
    lists["LL_Text"] = None

    #create set for LL Heads
    ll_head_indices = set()

    lists["Next"] = lists["Next"].fillna(-1).astype(int)
    lists["Prev"] = lists["Prev"].fillna(-1).astype(int)

    #convert LL columns of DF from float to int
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


'''
----------
Functions for preparing data to be used in a hierarchical editor
----------
'''

'''
Takes the linked list text dataframes, 
and adds path data to be used for constructing the Ag-Grid
'''
def prepare_data_for_hierarchy(df):
#sort the editted result, by the y value
    sorted_df = df.sort_values(by="y_min").reset_index(drop=True)

    #add a column for "type". This is just used to determine if a row is the root. make '' for all except root
    sorted_df['Type'] = ''

    #add a row that will be the static root of the hierarchy
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
        'Text': "ROOT",
        'Author': [None],
        'Title': [None],
        'Type': "ROOT",
    })

    #overwrite df var, assigning sorted df and also adding root row
    df = pd.concat([root_row, sorted_df], ignore_index=True)
    
    #for unique row identification, insert current row indices into the unused index col
    #df = df.reset_index()
    df = df.reset_index().rename(columns={'level_0': 'id'})

    #make string for populating list later on
    df['id'] = df['id'].astype(str)

    #add a path column. Make sure it's dtype=object 
    #this column is a list of the parents of this node (including the node itself)
    df['Path'] = [[] for _ in range(len(df))]
    
    #populate starting hierarchy, everything is child of Root
    df.at[0, 'Path'] = ['0']
    for i in range(1, len(df)):
        df.at[i, 'Path'] = ['0', df.at[i, 'id']]
    
    #POSSIBLE TODO: Apply algorithmic first pass to generate hierarchy

    #st.write("Edited result")
    #st.dataframe(df)

    df_subset = df[["id", "Path", "Text","Author", 'Title']]

    #st.write("Edited Subset")
    #st.dataframe(df_subset)

    return df_subset


'''
---------
Functions for creating a streamlit Ag Grid
---------
'''

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
        ? overNode.data.Path
        : [];


    console.log("New Parent Path :", newParentPath);
    console.log("Node Path: ", movingData.Path);
    var equal = true;
    if (newParentPath.length !== movingData.Path.length) {
        equal = false;  
    } else {
        newParentPath.forEach(function (item, index) {
            console.log("Comparing :");
            console.log(item, movingData.Path[index]);
                      
            if (movingData.Path[index] !== item) {
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
        var oldPath = node.data.Path;
        var fileName = oldPath[oldPath.length - 1];
        var newChildPath = newParentPath.slice();
        newChildPath.push(fileName);

        node.data.Path = newChildPath;

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
    return data.Path;
} 
""")

getTitle = JsCode("""
function getDataTitle(params) {
    //console.log("PARAMS", params);
    //console.log(params.data.title);
                  
    //have the grouped column be by the title
    return params.data.Text;
}
""")

'''
Function for building an Ag-Grid from a dataframe
'''
def createGrid(db):
    print("IN GRID BUILDER")
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