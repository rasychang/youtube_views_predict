import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

#load trained model
model = joblib.load('model_random_forest.pkl')

#function to preprocess the input data
def preprocess(data):


    return data

#function to predict views
def predict_views(input_data):
    processed_data = preprocess(input_data)
    prediction = model.predict(processed_data)
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
        input_data = pd.DataFrame([[category_number, thumb_uploaded_file, duration, videoPublishingDate]],
                                  columns=['category_number', 'thumb_uploaded_file', 'duration', 'videoPublishingDate'])
        
        prediction = predict_views(input_data)
        st.write(f"Your Youtube Video's Predicted Views: {prediction[0]}")