# Load libraries
import streamlit as st
import spacy_streamlit
import numpy as np
import pandas as pd
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


# fonction 

def data_cleaning(strings):

    strings = strings.lower().strip()

    strings = strings.replace('\'',' ')
    strings = strings.replace('/',' ')

    strings = re.sub(r'[^\w\s]', ' ', strings)

    # Remplacer les caractères spéciaux par un espace
    text_normalized = re.sub('[^A-Za-z ,éêèîôœàâ]+', ' ', strings)

    return text_normalized



def standardization(x):
    dico = {"rg": "rouge","rges" : "rouge","rge": "rouge", "rse": "rose" ,"rs" : "rose", "ch": "chardonnay","bl": "blanc", "bdx": "Bordeaux", "vdt": "vin de table", 'vdp': "vin de pays","blc": "blanc", "bib": "bag in box"}
    liste = x.split(' ')
    for i in range(len(liste)) :
      if liste[i] in dico.keys():
        liste[i] = dico[liste[i]]
    return ' '.join(liste)


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


