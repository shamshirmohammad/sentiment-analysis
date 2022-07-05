import random
from textwrap import wrap

import joblib
import pandas as pd

from model import get_model


def get_sentiment(txt):
    sentiment, confidence, sentiment_proba_dict = model.predict(txt)
    return sentiment


model = get_model()
df = pd.read_csv('../data/IMDB Dataset.csv')

idx = random.randint(0, len(df))
txt, target = df.loc[idx]
sentiment = get_sentiment(txt)
wrapped_text = "\n".join(wrap(txt))

print(f'Random index: {idx}')
print(f'Review text:\n{wrapped_text}\n')
print(f"Actual    : {target}")
print(f"Predicted : {sentiment}")
