import random
from textwrap import wrap

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
    vectorizer = joblib.load('data/vectorizer.joblib')
    model = joblib.load('data/logreg.joblib')
    return vectorizer, model

@st.cache(show_spinner=False)
def load_df():
    return pd.read_csv('data/IMDB Dataset.csv')

def get_sentiment(txt):
    txt_tf = vectorizer.transform([txt])
    pred = model.predict(txt_tf)[0]
    sentiment = class_names[pred]
    return sentiment

vectorizer, model = load_model()
df = load_df()
class_names = ['negative', 'positive']

st.set_page_config(
    page_title="Sentiment Analysis",
    initial_sidebar_state="expanded",
)

# title and description

st.markdown("<h1 style='text-align: center; color: red;'>IMDb Review Sentiment Analysis</h1>", unsafe_allow_html=True)
with st.form("input_form"):
    txt = st.text_area("Type your review text!")

    submitted = st.form_submit_button(label='Analyze')
    randomized = st.form_submit_button(label='Random review!')

if submitted:
    sentiment = get_sentiment(txt)
    st.subheader("""Your review is ...""")
    if sentiment == 'positive':
        st.success(f"✅ {sentiment.title()} ✅")
    else:
        st.error(f"❌ {sentiment.title()} ❌")

if randomized:
    idx = random.randint(0, len(df))
    txt, target = df.loc[idx]
    sentiment = get_sentiment(txt)
    # wrapped_txt = "\n".join(wrap(txt))

    st.subheader('Review text:')
    st.write(f'{txt}', unsafe_allow_html=True)
    st.subheader(f'Predicted sentiment ...')
    if sentiment == 'positive':
        st.success(f"✅ {sentiment.title()} ✅")

    else:
        st.error(f"❌ {sentiment.title()} ❌")

    if sentiment != target:
        st.error(f"But the actual review was **{target.upper()}** ...")
