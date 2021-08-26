import string
from spacy.lang.en.stop_words import STOP_WORDS

MODELS = ["sshleifer/distilbart-cnn-12-6", "sshleifer/distilbart-xsum-12-3", "google/pegasus-xsum"]
CORPUS_GOT_REVIEWS = "GameOfThrones_Reviews.csv"
CORPUS_GOT_REVIEWS_CLEAN = "GameOfThrones_Reviews_clean.csv"
STREAMLIT_COLOR_SUBTITLE = "#2471A3"
STREAMLIT_COLOR_TITLE = "#154360"
PUNCTUATIONS = string.punctuation
STOPWORDS = list(STOP_WORDS)