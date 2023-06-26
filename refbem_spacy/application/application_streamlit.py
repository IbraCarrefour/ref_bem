
from classes.function_class import *
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(14.7,10.27)})

# config

hide_streamlit_style = """
<style>

footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.write("Cette page a été créée par :")
st.sidebar.write("COULIBALY")
###


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
default_text = st.text_input(label= "Enter text to analyze.", value= "Véritable Petit beurre - LU - 12 X 3")
# Custom SpaCy Model
custom_model = spacy_streamlit.load_model('refbem_spacy/output/spacy_textcat/model-best')
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