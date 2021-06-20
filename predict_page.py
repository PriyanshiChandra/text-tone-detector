import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib as plt
import re
from data_load import redirect

def show_predict_page():
    st.title("Text tone detector")
    st.subheader("Enter your text and we will tell you how it will sound to the readers")
    text=st.text_area("Enter your text here")

    ok = st.button("Check Tone")

    if ok:
        ans = redirect(text)
        if ans == 0:
            st.write("""### Your text will sound negative to readers """)
        else:
            st.write("""### Your text will sound positive to readers """)