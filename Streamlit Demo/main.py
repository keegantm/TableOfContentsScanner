import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import datetime
import numpy as np
import requests
import uuid
import json


#https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/9a8503b806ef3237f2b6c21cf2ad4dc97dec568c/20_cell_renderer_class_example.py 
#good for buttons within the DF

#https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/9a8503b806ef3237f2b6c21cf2ad4dc97dec568c/70_nested_grids.py
#FOR HIERARCHY USE MASTER/DETAIL

#https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/9a8503b806ef3237f2b6c21cf2ad4dc97dec568c/82_Handling_Grid_events.py
#displays how to use event handlers. Could use to change hierarchy

#https://www.ag-grid.com/javascript-data-grid/grid-options/ RESEARCH row drag
#NOTE: If we use GroupBy instead of master/detail, can just drag between the hierarchies
#Alternative: override and use the use states to define custom dragging behavior

#NOTE: https://www.ag-grid.com/javascript-data-grid/tree-data/
#This could be perfect, a more flexible hierarchy. Enables dragging between hierarchies

# Define the Streamlit containers
header = st.container()

with header:
    st.title("StreamLit Demo")
    st.text("My test application, to determine if Streamlit is a good fit for my project")

#read in the JSON. Our CV algo would process a video/image, try to turn it into JSON in a certain format, and would be used here
with open('contents.json', 'r') as file:
    data = json.load(file)
    #df = pd.read_json(file)
    df = pd.DataFrame(data)
file.close()

df["subChapters"] = df["subChapters"].apply(lambda x: pd.json_normalize(x))


gridOptions = {
    # enable Master / Detail
    "masterDetail": True,
    "rowSelection": "single",
    "rowDragManaged": True,
    "rowDragEntireRow": True,

    # the first Column is configured to use agGroupCellRenderer
    "columnDefs": [
        {
            "field": "chapter",
            "cellRenderer": "agGroupCellRenderer",
        },
        {"field": "author"},
        {"field": "pageCount"},
    ],
    "defaultColDef": {
        "flex": 1,
    },
    # provide Detail Cell Renderer Params
    "detailCellRendererParams": {
        # provide the Grid Options to use on the Detail Grid
        "detailGridOptions": {
            "rowSelection": "single",
            "rowDragManaged": True,
            "rowDragEntireRow": True,
            "pagination": True,
            "paginationAutoPageSize": True,
            "columnDefs": [
                {"field": "subChapter"},
                {"field": "title"},
                {"field": "pageCount"},
            ],
            "defaultColDef": {
                "sortable": True,
                "flex": 1,
            },
        },
        # get the rows for each Detail Grid
        "getDetailRowData": JsCode(
            """function (params) {
                params.successCallback(params.data.subChapters);
        }"""
        ),
        
    },
    "rowData": data
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

# tabs =  st.tabs(['Selected Rows','gridoptions','grid_response'])

# with tabs[0]:
st.write(r.selected_rows_id)
