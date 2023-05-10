import streamlit as st
import pandas as pd

"## Classification basée sur les données apriori"

def barcode():
    barcod = st.text_input("Barcode" , value = 1, key = "barcode")
    return barcod

sparse_matrix = pd.read_csv('data.csv', sep = ',')
sparse_matrix.astype({'barcode': str})
" Tableau des probabilités"
st.dataframe(sparse_matrix)
def predict(barcode):
  return sparse_matrix.loc[sparse_matrix.barcode == barcode]

barcod = barcode()
if barcod :
   "codif possible"
   st.dataframe(predict(barcod)[["barcode", "classe", "Probabilité"]].sort_values(by=['Probabilité']))