import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import numpy as np
from tensorflow.keras.applications import VGG16
from PIL import Image

#load trained model
model = joblib.load('model/model_random_forest.pkl')

def process_image(image):
    # Open the uploaded image and perform preprocessing
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image to match the input size of the model
    img_array = np.array(img)
    
    # Process the image further if necessary
    normalized_image = img_array / 255.0
    
    return normalized_image

def extract_image_features(image_path):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    processed_image = process_image(image_path)
    if processed_image is not None:
        features = model.predict(np.expand_dims(processed_image, axis=0))
    return np.array(features.flatten())

#function to predict views
def predict_views(input_data):
    reshaped_arr = input_data.reshape(1, -1)
    prediction = model.predict(reshaped_arr)
    return prediction


#UI
st.title('YouTube Video Info Submition')

#input form
with st.form("video_info_form"):
    channelName = st.text_input("Channel Name")
    videoTitle = st.text_input("Video Title")
    categories = [
        ("Film & Animation", 1), 
        ("Autos & Vehicles", 2),
        ("Music", 10),
        ("Pets & Animals", 15),
        ("Sports", 17),
        ("Short Movies", 18),
        ("Travel & Events", 19),
        ("Gaming", 20),
        ("Videoblogging", 21),
        ("People & Blogs", 22),
        ("Comedy", 23),
        ("Entertainment", 24),
        ("News & Politics", 25),
        ("Howto & Style", 26),
        ("Education", 27),
        ("Science & Technology", 28),
        ("Nonprofits & Activism", 29),
        ("Movies", 30),
        ("Anime/Animation", 31),
        ("Action/Adventure", 32),
        ("Classics", 33),
        ("Documentary", 35),
        ("Drama", 36),
        ("Family", 37),
        ("Foreign", 38),
        ("Horror", 39),
        ("Sci-Fi/Fantasy", 40),
        ("Thriller", 41),
        ("Shorts", 42),
        ("Shows", 43),
        ("Trailers", 44)
    ]
    category = st.selectbox("Select a Category", options=[category[0] for category in categories])
    category_number = next(number for cat, number in categories if cat == category)


    thumb_uploaded_file = st.file_uploader("Upload a Thumbnail Image", type=['png', 'jpg', 'jepg'])
    duration = st.number_input("Duration (in seconds)", step=1, format="%d")
    tags = st.text_input("Tags (comma-separated)")
    description = st.text_area("Description")
    videoPublishingDate = st.date_input("Video Publishing Date").weekday()
    
    submitted = st.form_submit_button("Predict")
    

    
    if submitted:
        # print("category_number", category_number)
        # print("thumb_uploaded_file", thumb_uploaded_file)
        # print("duration", duration)
        # print("videoPublishingDate", videoPublishingDate)
        if not channelName or not videoTitle or not category_number or not thumb_uploaded_file or not duration or not tags or not description:
            st.error('All fields are required, please complete the input fields.')
        else:
            processed_image = extract_image_features(thumb_uploaded_file)
            features = np.array([category_number, videoPublishingDate, duration])
            input_data = np.concatenate([processed_image, features])
            prediction = predict_views(input_data)
            st.write(f"Your Youtube Video's Predicted Views: {prediction[0]}")