import streamlit as st
import pandas as pd

"## Classification basée sur les données apriori"

def barcode():
    barcod = st.text_input("Barcode" , value = 1, key = "barcode")
    return barcod

sparse_matrix = pd.read_csv('data.csv', sep = ',', dtype={1: str, 2: str})
#sparse_matrix.astype({'barcode': str, "classe": str})
sparse_matrix = sparse_matrix.convert_dtypes()
" Tableau des probabilités"

st.dataframe(sparse_matrix)
def predict(barcode):
  return sparse_matrix.loc[sparse_matrix.barcode == barcode]

barcod = barcode()
if barcod :
   "codif possible"
   st.dataframe(predict(barcod).sort_values(by=['Proportion']))