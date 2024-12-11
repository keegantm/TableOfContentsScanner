# TableOfContentsScanner

## Project Description

This prototype allows users to take a photograph of a table of contents, preprocess, and extract relevant text as determined by the user. The idea was inspired by mobile deposit features common in banking apps. Using basic preprocessing principles andpy tesseract for OCR, this full stack solution is a strong starting place for further feature development and refinement. 

## How it Works
1. Take a well lit photograph of the table of contents


2. Crop the image, focusing on the relevant text (less is more here!)
3. Draw a line to straighten the photograph
4. Preprocessing
5. Edit Extracted Information
6. Apply Hierarchy to Returned Author/Title information


## Contact Us
Keegan Moseley- keegantm@comcast.net
Ryan Moore - williamryanmoore@gmail.com

## Setup Instructions

1) **Install Dependencies**  
Install the conda enviroment in the .yaml file, or
download the following packages:
    - streamlit
    - cv2
    - numpy
    - PIL
    - streamlit_drawable_canvas
    - pandas
    - scipy
    - pytesseract Note: You will likely have to pip install this
    - st_aggrid

2) In the TableOfContentsScanner directory, use the command ```streamlit run main.py ``` to start the program
