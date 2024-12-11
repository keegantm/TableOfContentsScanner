# TableOfContentsScanner

## Project Description

This prototype allows users to take a photograph of a table of contents, preprocess, and extract relevant text as determined by the user. The idea was inspired by mobile deposit features common in banking apps. Using basic preprocessing principles andpy tesseract for OCR, this full stack solution is a strong starting place for further feature development and refinement. 

## How it Works
Take a well lit photograph of the table of contents
![Take a photo](Images/1.png)
Crop the image, focusing on the relevant text (less is more here!)
![Crop the photo](Images/2.png)
Draw a line to straighten the photograph
![Descew Photo](Images/3.png)   
Noise Reduction
![Noise Reduction](Images/4.png)
Text Blocks found using Erosion and Contouring
![Erosion and Contouring](Images/5.png)
Discovered Text Blocks
![Text Blocks](Images/6.png)
Edit Extracted Information
![Edit OCR Results](Images/7.png)
Apply Hierarchy to Returned Author/Title information
![Hierarchy](Images/8.png)
Final Structure
![Final](Images/9.png)



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
