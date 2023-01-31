import pandas as pd
import numpy as np
import streamlit as st
from googletrans import Translator
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

#Extract N-recommendations by score in order
def get_recommendations(N, ingredients):
    recipe_df = pd.read_csv("data/filipino_recipe_clean.csv")
    #Use TFIDF Model
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(recipe_df['ingredients_clean'])
    translator = Translator()
    text = translator.translate(ingredients, dest='en').text
    st.write(text)
    test = tf.transform([str(text)])
    test_cosine = cosine_similarity(tfidf_matrix, test)
    recommend_cosine = sorted(range(len(test_cosine)), key = lambda sub: test_cosine[sub])[-N:]

    #Create blank dataframe to load the recommendations 
    recommendations = pd.DataFrame(columns=['recipe', 'ingredients', 'url'])
    count = 0
    for i in recommend_cosine:
        food = '<p style="font-size: 24px; font-weight:800; text-align: left;">{}</p>'.format(recipe_df["food"][i])
        st.write(food, unsafe_allow_html=True)
        st.write('<p style="font-size: 18px; font-weight:500; text-align: left;"><b>Link: </b><a href="{}">{}</a></p>'.format(recipe_df["url"][i], recipe_df["url"][i]), unsafe_allow_html=True)
        ing = '<p style="font-size: 15px; font-weight:400; text-align: left;">{}</p>'.format(recipe_df["ingredients"][i])
        st.write(ing, unsafe_allow_html=True)
        st.markdown(" ")
        #recommendations.at[count, "recipe"] =  recipe_df["food"][i]
        #recommendations.at[count, "ingredients"] = recipe_df["ingredients"][i]
        #recommendations.at[count, "url"] = st.write('<p style="font-size: 15px; font-weight:500; text-align: left;">URL </p> <a href="{}">{}</a>'.format(recipe_df["url"][i], recipe_df["food"][i]), unsafe_allow_html=True)
        count += 1



def main():
    title = '<p style="font-size: 50px; font-weight:800; text-align: center;">What\'s your ulam, pare?: A filipino recipe recommendation system</p>'
    st.markdown(title, unsafe_allow_html=True)

    ingredients = st.text_input("Enter ingredients you would like to cook with (seperate ingredients with a comma)",
                                "onion, chorizo, chicken thighs, paella rice, frozen peas, prawns")

    if st.button("Give me recommendations!"):
        #ingredients
        recom = get_recommendations(5, ingredients)

main()