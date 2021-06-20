import streamlit as st  
from predict_page import show_predict_page

st.set_page_config(page_title='Text Tone Detector', page_icon = "favicon.jpeg")
show_predict_page()