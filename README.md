# IMDB Movie Review Sentiment Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imdb-review-sentiment-1.herokuapp.com/)

## Built with
<code><img height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="40" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="40" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="40" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/image4.jpg"></code>

<code><img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png"></code>
<code><img height="40" src="https://miro.medium.com/max/691/0*xXUYOs5MWWenxoNz"></code>
<code><img height="40" src="https://image4.owler.com/logo/hugging-face_owler_20191218_073707_original.png"></code>


## Summary
- This project was developed to test the capability of machine learning models in text classification (an NLP task), specifically sentiment analysis in this case.
- A **DistilBert** transformer model was built using `PyTorch` with the `HuggingFace` `transformers` library, however, it was quite a large model, therefore it is not used to deploy into the Web App for inference purposes. A **Logistic Regression** model was used instead as it is very lightweight and it was able to achieve similar accuracy in this case(**around 90%**). You may refer to the `analysis_modelling.ipynb` notebook for the entire process of building the models.
- You can also refer to the `inference.py` in the `src` folder to use the `DistilBert` model to predict random review sentiments. But before that, you need to download the trained `DistilBert` model (`best_model_state.bin`) from the [release page](https://github.com/ansonnn07/IMDB-Review-Sentiment-Analysis/releases/tag/1.0) first, and put it into the `data` folder for the script to work.


- **The Web App** was built by utilizing the capability of machine learning to classify movie reviews into either negative or positive sentiment.
- The movie review data was downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), and it was collected from IMDB movie reviews.

## Web Application
The Web App is accessible [here](https://imdb-review-sentiment-1.herokuapp.com/) which you can directly see all the visualizations made.

## Acknowledgement
- Link to the Kaggle dataset: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Huge thanks to the impressive library built by [huggingface](https://github.com/huggingface/transformers)