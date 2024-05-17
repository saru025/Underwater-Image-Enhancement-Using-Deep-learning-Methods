import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import from_pretrained_keras

# Load the model
model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

st.title("UNDER WATER IMAGE ENHANCEMENT")

uploaded_image = st.file_uploader("Choose a low-light image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    low_light_img = Image.open(uploaded_image).convert('RGB')
    st.image(low_light_img, caption="Input Low-Light Image", use_column_width=True)

    # Preprocess the image for model prediction
    low_light_img = low_light_img.resize((256, 256), Image.NEAREST)
    image_array = img_to_array(low_light_img)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Perform model inference to enhance the low-light image
    output = model.predict(image_array)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape((np.shape(output_image)[0], np.shape(output_image)[1], 3))
    output_image = np.uint8(output_image)

    st.image(Image.fromarray(output_image, 'RGB'), caption="Enhanced Image", use_column_width=True)