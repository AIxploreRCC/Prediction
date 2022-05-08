import pandas as pd
import matplotlib.pyplot as plt
import requests
from joblib import load
import io
import streamlit as st
import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (concordance_index_censored,
                            cumulative_dynamic_auc)
from sklearn.pipeline import make_pipeline


st.set_page_config(page_title="RCC Prognostic Nomogram", page_icon="🐞", layout="centered")
st.title("🐞 RCC Prognostic Nomogram")
st.markdown ("A post-operative prediction model which provides a comprehensive review of expected oncological outcomes in patient with renal cell carcinoma")
st.sidebar.image("https://urologie-rennes.fr/wp-content/uploads/2020/12/logo-02ai.svg", use_column_width=True)
st.sidebar.header("Service d'urologie Rennes")

st.markdown ("**Enter Your Information**")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://share.streamlit.io/aixplorercc/test/main/Database.py" target="_blank">Urologie Rennes</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.youtube.com/results?search_query=urologie+rennes" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/ZineEddineKhene" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

def user_input():
    Tumorsize=st.slider("Tumor size", min_value = 0, max_value = 50, value = 80)
    Preoperativehemoglobin=st.slider("Preoperative hemoglobin", min_value = 5, max_value = 20, value = 10)
    Vascularinvasion=st.selectbox("Vascular invasion", options = ["0", '1'])
    Perinephricfatinvasion=st.selectbox("Perinephric fat invasion", options = ["0", '1'])
    Nodalinvolvement=st.selectbox("Nodal involvement", options = ["0", '1'])
    Coagulativenecrosis=st.selectbox("Coagulative necrosis", options = ["0", '1'])
    Sarcomatoidfeatures=st.selectbox("Sarcomatoid features", options = ["0", '1'])
    ECOGperformancestatus=st.selectbox("ECOG performance status", options = ["0", '1', '2', '3'])
    Nucleargrade=st.selectbox("Nuclear grade", options = ["1", '2', '3', '4'])
    Histology=st.selectbox("Histology", options = ["1", '2', '3', '4'])
    dff={'Tumor size':Tumorsize,
         'Preoperative hemoglobin':Preoperativehemoglobin,
         'Vascular invasion':Vascularinvasion,
         'Perinephric fat invasion':Perinephricfatinvasion,
         'Nodal involvement':Nodalinvolvement,
         'Coagulative necrosis':Coagulativenecrosis,
         'Sarcomatoid features':Sarcomatoidfeatures,
         'ECOG performance status': ECOGperformancestatus,
         'Nuclear grade':Nucleargrade,
         'Histology':Histology
         
    }
    resultat=pd.DataFrame(dff,index=[0])
    return resultat

df=user_input()


st.subheader('Paramètres pré opératoire')
st.write(df.astype('object'))



# Loading the Saved Model
model = load("rsf2.joblib")


