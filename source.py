import pandas as pd
import requests
import io
import streamlit as st
import sklearn
from sklearn.linear_model import LogisticRegression


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

username = 'AI_xplore'
token = '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'
url = 'https://raw.githubusercontent.com/AIxploreRCC/RCC/main/complications.csv?token=GHSAT0AAAAAABRBZBPW7ZLQ4JWR5QNMHMIYYPWOWVA'
github_session = requests.Session()
github_session.auth = (username, token)

@st.cache
def load_data(nrows):
    download = github_session.get(url).content
    data = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return data

data= load_data(1000)



def user_input():
    Age=st.slider("Age", min_value = 21, max_value = 100, value = 80)
    Sexe=st.selectbox("Sexe", options = ["0", '1'])
    BMI=st.slider("BMI", min_value = 15, max_value = 40, value = 60)
    ECOG=st.selectbox("ECOG", options = ["0", '1', '2', '3'])
    dff={'Age':Age,
          'Sexe':Sexe,
          'BMI':BMI,
          'ECOG': ECOG
    }
    resultat=pd.DataFrame(dff,index=[0])
    return resultat

df=user_input()



st.subheader('Paramètres pré opératoire')
st.write(df.astype('object'))



data= load_data(100)

from sklearn.impute import KNNImputer
import numpy as np
imputer2= KNNImputer(missing_values=np.nan, n_neighbors=1)
data2 = pd.DataFrame(imputer2.fit_transform(data),columns = data.columns)





X=data2[['Age','Sexe','BMI', 'ECOG']]
y=data2['Complication']

clf=LogisticRegression()
LR=clf.fit(X, y)

key_thresh = 0.6    

pred_prob_adjusted_array = LR.predict_proba(df)
pred_prob_adjusted = round(pred_prob_adjusted_array[0,1],2)

# assign class
if pred_prob_adjusted < key_thresh :
    pred_class = "Positive"
    biopsy = "No"
else:
    pred_class = "Negative"
    biopsy = "Yes"

st.subheader("Risque de complication:")

st.write("Probability of complications", round(1- pred_prob_adjusted,2))
st.write('Recommend for avoid surgery:', biopsy)
