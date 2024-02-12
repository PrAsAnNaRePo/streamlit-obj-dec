import streamlit as st
from transformers import pipeline
import cv2
import os

def draw_boxes(image_path):
    outputs = obj_pipe(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    font_scale = max(img.shape) / 1000
    for output in outputs:
        xmin = output['box']['xmin']
        ymin = output['box']['ymin']
        xmax = output['box']['xmax']
        ymax = output['box']['ymax']

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        label = output['label']
        score = output['score']

        text = f"{label}: {score:.2f}"
        cv2.putText(img, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    return img

st.title("Object Detection App")
st.write("This app detects objects in your images")
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
obj_pipe = pipeline(task='object-detection', model='hustvl/yolos-base')
if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    filename = "uploaded_image.jpg"
    with open(filename, "wb") as f:
        f.write(image)
    with st.spinner('Wait for it...'):
        image = draw_boxes(filename)
    st.image(image, caption='Detected Image', use_column_width=True)
    os.remove(filename)
