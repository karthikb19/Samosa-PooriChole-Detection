from fastai.vision import open_image, load_learner, image, torch
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO

# App title
st.title("Poori Chole vs Samosa")


def predict(img, display_img):
    st.image(display_img, use_column_width=True)

    with st.spinner('Wait for it...'):
        time.sleep(5)

    model = load_learner('model/images/')
    pred_class = model.predict(img)[0]
    pred_prob = (torch.max(model.predict(img)[2]).item() * 100)

    if str(pred_class) == 'poorichole':
        st.success("This is Poori Chole. Probability -> " + str(pred_prob) + '%.')
    else:
        st.success("This is Samosa. Probability -> " + str(pred_prob) + '%.')


option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    test_images = os.listdir('model/testimages/')
    test_image = st.selectbox(
        'Select image:', test_images)

    file_path = 'model/testImages/' + test_image
    img = open_image(file_path)
    display_img = mpimg.imread(file_path)
    predict(img, display_img)

else:
    url = st.text_input("Input a url:")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img)  # Image to display

            img = pil_img.convert('RGB')
            img = image.pil2tensor(img, np.float32).div_(255)
            img = image.Image(img)

            predict(img, display_img)

        except:
            st.text("URL not valid!")
