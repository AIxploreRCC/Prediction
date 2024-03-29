import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from joblib import load
import pickle
import io
import streamlit as st
import sksurv
from sksurv.ensemble import RandomSurvivalForest
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


# Loading the Saved Model

model = load('rsfp.joblib')

# Define choices and labels for feature inputs
CHOICES = {0: 'No', 1: 'Yes'}

def format_func_yn(option):
    return CHOICES [option]
  
Histology2 = { 
    4: "Other",
    3: "chRCC",
    2: "pRCC",
    1: "cRCC",
    }   
  

# get inputs

Tumor_size=st.slider("Tumor size", min_value = 1, max_value = 20, value = 10)
Preoperative_hemoglobin=st.slider("Preoperative hemoglobin", min_value = 5, max_value = 20, value = 10)
Vascular_invasion=st.selectbox("Vascular invasion", options=list(CHOICES.keys()), format_func=format_func_yn, index=1)
Perinephric_fat_invasion=st.selectbox("Perinephric fat invasion", options=list(CHOICES.keys()), format_func=format_func_yn, index=1)
Nodal_involvement=st.selectbox("Nodal involvement", options=list(CHOICES.keys()), format_func=format_func_yn, index=1)
Coagulative_necrosis=st.selectbox("Coagulative necrosis", options=list(CHOICES.keys()), format_func=format_func_yn, index=1)
Sarcomatoid_features=st.selectbox("Sarcomatoid features", options=list(CHOICES.keys()), format_func=format_func_yn, index=1)
ECOG_performance_status=st.selectbox("ECOG performance status", options = ["0", '1', '2', '3'])
Nuclear_grade=st.selectbox("Nuclear grade", options = ["1", '2', '3', '4'])
Histology=st.selectbox("Histology", options= (4, 3, 2, 1), format_func=lambda x: Histology2.get(x),
    )   
    
  
dff = pd.DataFrame (
  {
          'Tumor_size':[Tumor_size],
          'Preoperative_hemoglobin':[Preoperative_hemoglobin],
          'Vascular_invasion':[Vascular_invasion],
          'Perinephric_fat_invasion':[Perinephric_fat_invasion],
          'Nodal_involvement':[Nodal_involvement],
          'Coagulative_necrosis':[Coagulative_necrosis],
          'Sarcomatoid_features':[Sarcomatoid_features],
          'ECOG_performance_status':[ECOG_performance_status],
          'Histology':[Histology],
          'Nuclear_grade': [Nuclear_grade]
    }
)

#Preprocessing

dff["Histology"]=pd.Categorical(dff["Histology"],ordered=False)
dff["ECOG_performance_status"]=pd.Categorical(dff["ECOG_performance_status"],ordered=True)
dff["Histology"]=pd.Categorical(dff["Histology"],ordered=False)
dff["Nodal_involvement"]=pd.Categorical(dff["Nodal_involvement"],ordered=False)
dff["Nuclear_grade"]=pd.Categorical(dff["Nuclear_grade"],ordered=False)
dff["Perinephric_fat_invasion"]=pd.Categorical(dff["Perinephric_fat_invasion"],ordered=False)
dff["Coagulative_necrosis"]=pd.Categorical(dff["Coagulative_necrosis"],ordered=False)
dff["Sarcomatoid_features"]=pd.Categorical(dff["Sarcomatoid_features"],ordered=False)
dff["Vascular_invasion"]=pd.Categorical(dff["Vascular_invasion"],ordered=False)


scaling_cols = [c for c in dff if dff[c].dtype.kind in ['i', 'f']]
cat_cols = [c for c in dff if dff[c].dtype.kind not in ["i", "f"]]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
preprocessor = ColumnTransformer([('cat-preprocessor', OrdinalEncoder(), cat_cols),
    ('standard-scaler', StandardScaler(), scaling_cols)], remainder='passthrough', sparse_threshold=0)

    
surv = model.predict(dff)

st.write(surv)

st.markdown ("Disease Free Survival Probability")

pred_surv = model.predict_survival_function(dff, return_array=True)


fig, ax = plt.subplots()
for i, s in enumerate(pred_surv):
    plt.step(model.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")

# Tick labels
plt.xlim(0, 60)
x_positions = (0, 12, 24, 36, 48, 60)
plt.xticks(x_positions, rotation=0)
plt.legend()
plt.grid(True)

st.pyplot (fig)
         

st.markdown ("Disease Free Survival Probability")

times = np.arange(6, 61, 6)

rsf_surv_prob = np.row_stack([
    fn(times)
    for fn in model.predict_survival_function(dff, return_array=False)
])

DFS_table = pd.DataFrame(rsf_surv_prob, columns = times)

st.write (DFS_table)
