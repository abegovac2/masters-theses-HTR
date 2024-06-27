import streamlit as st
from PIL import Image
import os
from run_pipeline import TextDetectionPipeline

# Set up a directory to store uploaded images
UPLOAD_DIR = "uploads"
td_pipeline = TextDetectionPipeline()

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


def clear_helper_dir(helper_dir="./helper_dir"):
    for f in os.listdir(helper_dir):
        os.remove(f"{helper_dir}/{f}")


# Title of the app
st.title("Image Upload and History App")

# Initialize session state to store the selected image file name
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Handle file upload
if uploaded_file is not None:
    clear_helper_dir()
    # Save uploaded file to the uploads directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Image uploaded successfully!")
    # Update the selected image to the newly uploaded image
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
            clear_helper_dir()
            update_selected_image(file)
else:
    st.sidebar.write("No images uploaded yet.")

# Display the selected image
if st.session_state.selected_image:
    image = Image.open(os.path.join(UPLOAD_DIR, st.session_state.selected_image))
    st.image(image, caption=st.session_state.selected_image, use_column_width=True)
    # if st.button("Process image", on_click=lambda: st.success("pressed")):
    if st.button("Process image"):
        regions = [5]
        result = td_pipeline._execute(
            f"./{UPLOAD_DIR}/{st.session_state.selected_image}", regions
        )
        result = list(zip(regions, result))
        for i, res in result:
            st.subheader(f"Region {i}")
            for r in res:
                st.image(r[0], r[1], use_column_width=True)
