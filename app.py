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
st.title('Sentiment Analysis')
st.markdown('''
- This Web App was built by utilizing the capability of machine learning to classify
movie reviews into either negative or positive sentiment.
- The movie review data was downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews),
and it was collected from IMDB movie reviews.
- Not every single review in the data was labeled correctly, so there may be incorrect predictions
when you try to run the sentiment analysis in the Web App here.
- And unfortunately the movie names were not included in the dataset, sorry about that 😔


- App built by [Anson](https://www.linkedin.com/in/ansonnn07/)
- Built with `Python`, using `streamlit`, `sklearn`, `pytorch`, `pandas`, `numpy`

**Links**: [GitHub](https://github.com/ansonnn07/IMDB-Review-Sentiment-Analysis), 
[LinkedIn](https://www.linkedin.com/in/ansonnn07/),
[Kaggle](https://www.kaggle.com/ansonnn/code)
''')


st.markdown("""
## **How to Use the App**
You have **two options**:
- Type your **own review** into the text box and click the **"Analyze"** button to test the predicted result!
- Or you can click the **"Random review"** to randomly generate a review from the database and let the model
predict the sentiment! But bear in mind that there may be incorrect predictions and it will let you know
if the prediction is incorrect.

Perhaps you can try to spam the **"Random Review"** button until you get an incorrect prediction 
from the model, but beware that the model has an accuracy of 90% out of 50k reviews! 😜
""")

st.markdown('---')


with st.form("input_form"):
    txt = st.text_area("Type your review text!")

    submitted = st.form_submit_button(label='Analyze')
    randomized = st.form_submit_button(label='Random review!')


if submitted:
    sentiment = get_sentiment(txt)
    st.subheader("""Your review is ...""")
    if sentiment == 'positive':
        st.success(f"{sentiment.title()} ! Thank you 😊")
    else:
        st.error(f"{sentiment.title()} ... Aww 😔")

if randomized:
    idx = random.randint(0, len(df))
    txt, target = df.loc[idx]
    sentiment = get_sentiment(txt)
    # wrapped_txt = "\n".join(wrap(txt))

    st.subheader('Review text:')
    st.write(f'{txt}', unsafe_allow_html=True)
    st.subheader(f'Predicted sentiment ...')
    if sentiment == 'positive':
        st.success(f"{sentiment.title()} ! 😊")

    else:
        st.error(f"{sentiment.title()} ... 😔")

    if sentiment != target:
        st.error(f"But the actual review was **{target.upper()}** ...")

# st.image(Image.open(''))
