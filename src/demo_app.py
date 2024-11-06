import streamlit as st
from PIL import Image, ImageDraw
import os
from rest.dependencies import (
    get_groq_client,
    get_extraction_services,
)
from rest.api.text_extraction_api import line
import asyncio
from fastapi import UploadFile
import io

# Apply CSS to make Streamlit layout full width
st.markdown(
    """
    <style>
         [data-testid="stAppViewBlockContainer"] {
            max-width: 100rem;
            margin-left: auto;
            margin-right: auto;
        }
        [data-testid="column"] {
            height: 700px;
            overflow-y: scroll;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up a directory to store uploaded images
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Title of the app
st.title("Text Extraction Demo App")

# Initialize session state to store the selected image file name
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Handle file upload
if uploaded_file is not None:
    # Save uploaded file to the uploads directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Image uploaded successfully!")
    st.session_state.selected_image = uploaded_file.name

# List all uploaded files
uploaded_files = os.listdir(UPLOAD_DIR)

# Sidebar for showing history of uploaded images
st.sidebar.title("Uploaded Images History")

# Function to update the selected image
def update_selected_image(file):
    st.session_state.selected_image = file

# Display clickable image titles in the sidebar
if uploaded_files:
    for file in uploaded_files:
        if st.sidebar.button(file, on_click=update_selected_image, args=(file,)):
            update_selected_image(file)
else:
    st.sidebar.write("No images uploaded yet.")

# Display the selected image with processing options
if st.session_state.selected_image:
    # Load the original image
    image_path = os.path.join(UPLOAD_DIR, st.session_state.selected_image)
    original_image = Image.open(image_path)

    # Display the original image, bounding boxes, and detected words in a 3-column layout
    col1, col2, col3 = st.columns([3, 3, 4])  # Adjusted for full-width

    # Column 1: Display the original image
    with col1:
        st.header("Original Image")
        st.image(original_image, use_column_width=True)

    # Process image if button is clicked
    if st.button("Process Image"):
        img = None
        with open(image_path, "rb") as r:
            img = r.read()

        # Call the text extraction API
        res = asyncio.run(
            line(
                UploadFile(
                    file=io.BytesIO(img), filename=st.session_state.selected_image
                ),
                include_image=True,
                enhance_with_llm=True,
                extraction_services=get_extraction_services(),
                llm_client=get_groq_client(),
            )
        )

        # Create a copy of the image to draw bounding boxes
        bbox_image = original_image.copy()
        draw = ImageDraw.Draw(bbox_image)

        detected_words = {}  # Store detected words for display in col3

        # Draw bounding boxes on detected regions
        for r, reg in res.regions.items():
            detected_region_words = []
            for det in reg.detections:
                # Extract bounding box coordinates
                reg_tl_x = reg.bounding_box.top_left.x
                reg_tl_y = reg.bounding_box.top_left.y

                top_left = (
                    reg_tl_x + det.bounding_box.top_left.x,
                    reg_tl_y + det.bounding_box.top_left.y,
                )
                bottom_right = (
                    reg_tl_x + det.bounding_box.bottom_right.x,
                    reg_tl_y + det.bounding_box.bottom_right.y,
                )
                # Draw rectangle on the image
                draw.rectangle([top_left, bottom_right], outline="red", width=4)
                detected_region_words.append(
                    (det.line_image.image, det.text)
                )  # Add detected text to list
            detected_words[r] = detected_region_words

        # Column 2: Display the image with bounding boxes
        with col2:
            st.header("Bounding Boxes")
            st.image(bbox_image, use_column_width=True)

        # Column 3: Display a list of detected words
        with col3:
            st.header("Detected Words")
            if detected_words:
                items = [
                    (region, detected_region_words)
                    for region, detected_region_words in detected_words.items()
                ]
                items = list(sorted(items, key=lambda el: el[0]))
                for region, detected_region_words in items:
                    st.subheader(f"Region {region}")
                    for word_image, word in detected_region_words:
                        st.image(word_image, word, use_column_width=True)
            else:
                st.write("No words detected.")
