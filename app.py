# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Product Brand Identification on Retail Stores Shelves",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Product Brand Identification on Retail Stores Shelves")
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Logo.jpeg", width=250, height=160)
st.sidebar.image(my_logo)

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Brand Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Brand Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,hide_conf=True
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        names = model.names

                        cla = []

                        for c in res[0].boxes.cls:
                            cla.append(names[int(c)])

                        arrxy=res[0].boxes.xyxy
                        coordinates = np.array(arrxy)


                        x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2

                        y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2

                        midpoints = np.column_stack((x_coords, y_coords))

                        rounded_n_sorted_arr = np.round(midpoints[midpoints[:, 1].argsort()]).astype(int)

                        count=1
                        objects=0
                        group_sizes = []

                        Obj = []

                        for i,j in zip(range(1, len(rounded_n_sorted_arr)),cla):

                            if(rounded_n_sorted_arr[i][1] - rounded_n_sorted_arr[i-1][1] > 130 ):
                                group_sizes.append(objects + 1)
                                count += 1
                                objects = 0
                                Obj.append(j)

                            else:
                                objects += 1

                        group_sizes.append(objects + 1)

                        rc = []

                        for i,j in zip(list(midpoints),cla):
                            k = list(i)
                            k.append(j)
                            gh = k
                            rc.append(gh)
                            
                        sorted_list = sorted(rc, key=lambda x: x[1])

                        big = []
                        for i in group_sizes:
                            count=0
                            li = []
                            al = []
                            for j in sorted_list:
                                count = count+1
                                li.append(j[2])
                                al.append(j)
                                if(count==i):
                                    break
                            for k in al:
                                sorted_list.remove(k)
                            big.append(li)
                            
                        data = big
                        # Initialize a dictionary to store class counts for each list
                        list_class_counts = []

                        # Iterate through each list in the data
                        for sublist in data:
                            # Create a dictionary to store class counts for the current list
                            class_counts = {}
                            
                            # Iterate through each element in the sublist
                            for item in sublist:
                                # Check if the class is already in the dictionary, if not, initialize it with 1
                                if item not in class_counts:
                                    class_counts[item] = 1
                                else:
                                    # If the class is already in the dictionary, increment the count
                                    class_counts[item] += 1
                            
                            # Append the class counts for the current list to the list_class_counts
                            list_class_counts.append(class_counts)
                            
                        for i, class_counts,j in zip(range(1,len(list_class_counts)+1), list_class_counts,group_sizes):
                            st.write(f"Shelf {i}:")
                            for class_name, count in class_counts.items():
                                st.write(f"         Brand: {class_name}, Count: {count}, Percentage: {round((count/j)*100,2)}")
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
