import time
import torch
import openai
import numpy as np
import torch.nn as nn
import streamlit as st

from sentence_transformers import SentenceTransformer, util
from src.engine import file_extract, get_similar_sentences, extract, openai_inference

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai.api_key = st.secrets["OPENAI_KEY"]

def text(text, tag='h1', align='left', color=None):
    assert tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'u', 'i', 'p'], "Invalid Tag"
    if color is None:
        st.markdown(f"<{tag} style='text-align: {align}'>{text}</{tag}>", unsafe_allow_html=True)
    else:
        st.markdown(f"<{tag} style='text-align: {align}; color: {color}'>{text}</{tag}>", unsafe_allow_html=True)

def header():
    st.title("GPT Search")
    st.markdown("Made with 🖤 by **[Tanay](https://twitter.com/serious_mehta)**")
    text("Upload a text file or Choose one sample text file and ask ChatGPT to answer queries from it!", "p")
    st.caption("Please Note: This is not a summarization app, it merely answers questions that can be found in the text corpus.")

def selectbox():
    """
    User can choose from two files to use
    Returns a list of individual sentences from either text file
    """
    option = st.selectbox(
        "Select a sample document",
        options=[
            "Impact of Technology on Society", 
            "The Benefits and Risks of Artificial Intelligence"
        ]
    )
    if option == "Impact of Technology on Society": file_name = "samples/sample_1.txt"
    else: file_name = "samples/sample_2.txt"
    # Open the file nonetheless of the display outcome
    with open(file_name, "r") as fl:
        data_actual = fl.readlines()
        # Remove newline character from line ends and only select non-empty strings as lines
        data = [x.replace('\n', '') for x in data_actual]
        data = [x for x in data if x]
    
    if st.button(f"Display '{option}.txt'"):
        display_txt_0 = text("“", tag="h2", align='left')
        display_txt_1 = st.caption(" ".join(data_actual))
        display_txt_2 = text("”", tag="h2", align='right')
        if st.button("Hide the text"):
            del display_txt_0, display_txt_1, display_txt_2
        return data, file_name
    else:
        return data, file_name
        
        
def file_upload():
    """
    User can upload a file of their choice. It should be a text file.
    Returns a list of individual sentences from the uploaded text file
    """
    file = st.file_uploader("Choose a text file")
    if file is not None:
        try:
            data = file.getvalue().decode('utf-8').splitlines()
            # Only select non-empty strings as lines
            data = [x for x in data if x]
            return data
        except:
            text(
                "Invalid file, please check that it is indeed a text (.txt) file and try uploading again!", 
                'strong', 
                'left', 
                'red'
            )

def option_choice():
    genre = st.radio(
        "Do you want to select a sample text file OR upload your own?",
        ('Select Sample text file', 'Upload my own')
    )
    if genre == "Select Sample text file": flag=False
    else: flag = True
    return flag
    
@st.cache
def get_model():
    return SentenceTransformer("all-mpnet-base-v2")

if __name__ == "__main__":
    # Emdash might cause troubles in some browsers
    st.set_page_config(
        page_title="GPT Search – By Tanay",
    )
    model = get_model()
    header()
    upload = option_choice()
    if upload is True:
        lines = file_upload()
        query = st.text_input("Enter your Query")
        if not lines: text("Please upload a file!", "p", color="red")
    else:
        lines, file_name = selectbox()
        if file_name == "samples/sample_1.txt":
            query = st.text_input("Enter your Query", "ex: How has technology transformed the way we work?")
        else:
            query = st.text_input("Enter your Query", "ex: How can AI be used in healthcare?")
    
    if lines:
        # Add a slider for top-k results and temperature
        # extra_param = st.checkbox("Extra Parameters")
        k_val, temp = 5, 0.2
        # if extra_param:
        #     st.write("If you don't know what any of this means, just leave them as it is.")
        #     cols_0, col_1 = st.columns([1, 1])
        #     with cols_0:
        #         k_val = st.slider('Select how many top-results to use?', 2, 10, 5)
        #         st.caption("Only select a value > 5 when you have a large text corpus.")
        #     with col_1:
        #         temp = st.slider('GPT-temperature', 0.01, 2.00, 0.20)
        #         st.caption("Higher the temperature, more creative but unstable results.")
        start = time.time()
        text_embeddings = file_extract(model, lines)
        query_embeddings = extract(model, lines)
        if st.button("GO!"):
            sentences = get_similar_sentences(lines, query_embeddings, text_embeddings, k=k_val)
            data = {'query': query, 'top_results': sentences}
            res = openai_inference(data, k=k_val, temperature=temp)
            time_taken = time.time() - start
            text(f"Results", "h5")
            st.caption(f"(took {time_taken:.4f} seconds)")
            st.write(res)