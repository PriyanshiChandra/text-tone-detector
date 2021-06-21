import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib as plt
import re
from data_load import prediction

def show_predict_page():
    st.title("Text tone detector")
    st.subheader("Enter your text and we will tell you how it will sound to the readers")
    text=st.text_area("Enter your text here")

    ok = st.button("Check Tone")

    if ok:
        ans = prediction(text)
        if ans == "Negative":
            st.markdown(":-1:")
            st.write("""### Your text will sound negative to readers. Please change the  text to sound positive """)
        else:
            st.markdown(":clap:")
            st.write("""### Good Job. Your text will sound positive to readers """)
        st.write("Feedback")
    
