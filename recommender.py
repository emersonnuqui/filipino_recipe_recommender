import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Set web page to wide layout 
st.set_page_config(layout="wide",initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

#Setting font for all text to Raleway
st.write("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100;0,200;0,300;0,400;0,600;0,700;0,800;1,300&display=swap');
    html, body, [class*="css"]  {
    font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True) 


def load_data():
    # Load the data for tracks and daily charts
    recipe_df = pd.read_csv('data/filipino_recipe_clean.csv')

    return recipe_df

def tfidf_matrix():
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['ingredients_clean'])
    text = 'ube, evaporated milk'
    tf_input = tf.transform([text])
    return tfidf_matrix, tf_input