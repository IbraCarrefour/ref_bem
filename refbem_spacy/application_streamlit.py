# Load libraries
import spacy
import streamlit as st
import spacy_streamlit
import numpy as np
import pandas as pd
import random
import re

import nltk
nltk.download('stopwords')

import plotly.express as px
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(14.7,10.27)})
from datetime import datetime

# config

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.write("Cette page a été créée par :")
st.sidebar.write("COULIBALY")
###


# fonction 

def data_cleaning(strings):

    strings = strings.lower().strip()

    strings = strings.replace('\'',' ')
    strings = strings.replace('/',' ')

    strings = re.sub(r'[^\w\s]', ' ', strings)

    # Remplacer les caractères spéciaux par un espace
    text_normalized = re.sub('[^A-Za-z ,éêèîôœàâ]+', ' ', strings)

    return text_normalized


dico = {"rg": "rouge","rges" : "rouge","rge": "rouge", "rse": "rose" ,"rs" : "rose", "ch": "chardonnay","bl": "blanc", "bdx": "Bordeaux", "vdt": "vin de table", 'vdp': "vin de pays","blc": "blanc", "bib": "bag in box"}

def standardization(x):
  liste = x.split(' ')
  for i in range(len(liste)) :
    if liste[i] in dico.keys():
      liste[i] = dico[liste[i]]
  return ' '.join(liste)



import nltk
from nltk.corpus import stopwords

def remove_stop_words(strings, stops):
    list_item_desc = strings.split(' ')
    cleaned_list = [word for word in list_item_desc if word not in stops]
    return ' '.join(cleaned_list)



def Pipeline(strings : str):
  strings = data_cleaning(strings)
  strings = standardization(strings)

  fr_stops = set(stopwords.words('french'))
  measurments = set(['oz','kg','g','lb','mg','l','cl','ml','tsp','tbsp',"cm","x", 'cte'])
  fr_stops.update(measurments)
  strings =  remove_stop_words(strings, fr_stops)

  return strings




st.header(":star: Text Classification with spacy")

"This application uses an initial example of item_desc classification for the 20 most represented class_desc in FRA + BEL."

if st.checkbox(label= "Display the class_desc used", value=False):
    """
    bières,
    vins,
    viande volaille libre service,
    traiteur plats cuisinés frais,
    europe,
    confiserie de sucre,
    sel poivre épices aides culinaires,
    confiserie de chocolat,
    autres fromages,
    déodorants,
    shampooings,
    pâtes alimentaires,
    biscuits sucrés,
    colorations teintures,
    access coiffure et de toilette,
    produits diététiques,
    soins du visage femme,
    maquillage,
    confiserie saisonnière,
    douches'
    """
default_text = st.text_input(label= "Enter text to analyze.", value= "nutella")
# Custom SpaCy Model
custom_model = spacy_streamlit.load_model('output/spacy_textcat/model-best')
if default_text :
    doc= custom_model(Pipeline(default_text))
    title = "Text"
    #spacy_streamlit.visualize_textcat(doc, title=title)
    names = list(doc.cats.keys())
    values = list(doc.cats.values())

    df = pd.DataFrame({"class_desc_fr": names, "score": values})

    fig = px.histogram(df, y = "class_desc_fr", x = "score", orientation="h")

    st.plotly_chart(fig)

    prediction = max(doc.cats, key=lambda key: doc.cats[key])
    confidence = str(np.round(doc.cats[prediction],2))
    st.header("Prediction: " + prediction)
    st.subheader("Confidence: " + confidence)