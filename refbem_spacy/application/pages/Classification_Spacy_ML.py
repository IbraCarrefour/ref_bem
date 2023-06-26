import streamlit as st
import spacy_streamlit
import numpy as np
import pandas as pd
from classes.function_class import *
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(14.7,10.27)})
import joblib
# config

st.set_page_config(
    page_title="4 most represented class_desc",
    page_icon=":shark:"
)

hide_streamlit_style = """
<style>

footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


###


st.header(":star: Text Classification with spacy and Machine Learning")

"This application uses an initial example of item_desc classification for the 4 most represented class_desc in FRA + BEL."

if st.checkbox(label= "Display the class_desc used", value=False):
    """
    vins,
    confiserie de sucre,
    maquillage,
    confiserie saisonnière,
    """
default_text = st.text_input(label= "Enter text to analyze.", value= "Véritable Petit beurre - LU - 12 X 3")
# Custom SpaCy Model
custom_model = spacy_streamlit.load_model('refbem_spacy/modele_ml/output/spacy_textcat/model-best')

model_name = st.sidebar.selectbox(label= "select model:", options= ("Spacy", "Linear regression", "Naive bayes"))

if default_text :
    if model_name == "Spacy":
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
    elif model_name == "Linear regression":
        filename = 'refbem_spacy/modele_ml/classifier_rl.pkl'
        loaded_model = joblib.load(filename)

        loaded_tv = joblib.load("refbem_spacy/modele_ml/TfidfVectorizer_transf.pkl")

        default_text = loaded_tv.transform([Pipeline_ml(default_text)])
        
        load_label = joblib.load("refbem_spacy/modele_ml/labelencoder_transf.pkl")

        "# Prediction :", load_label.inverse_transform(loaded_model.predict(default_text))[0]

    elif model_name == "Naive bayes":
        filename = 'refbem_spacy/modele_ml/classifier_nb.pkl'
        loaded_model = joblib.load(filename)

        loaded_tv = joblib.load("refbem_spacy/modele_ml/TfidfVectorizer_transf.pkl")

        default_text = loaded_tv.transform([Pipeline_ml(default_text)])
        
        load_label = joblib.load("refbem_spacy/modele_ml/labelencoder_transf.pkl")

        "# Prediction :", load_label.inverse_transform(loaded_model.predict(default_text))[0]