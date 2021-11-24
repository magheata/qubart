import string
from spacy.lang.en.stop_words import STOP_WORDS

CORPUS_GOT_REVIEWS = "GameOfThrones_Reviews.csv"
CORPUS_GOT_REVIEWS_CLEAN = "GameOfThrones_Reviews_clean.csv"

REV_SEASON = "Season"
REV_EPISODE = "Episode"
REV_REVIEW = "Episode recap"
REV_TITLE = "Episode title"

STREAMLIT_COLOR_SUBTITLE = "#2471A3"
STREAMLIT_COLOR_TITLE = "#154360"

SEPARATOR = "_"
REGEX_EOS = r"(?<!\..)[.?!]\s*"
PUNCTUATIONS = string.punctuation
STOPWORDS = list(STOP_WORDS)

OPTIONS_NER = ('Off', 'On')
OPTIONS_EMBEDDING = ("sent2vec", "doc2vec")
MODELS = ["sshleifer/distilbart-cnn-12-6", "sshleifer/distilbart-xsum-12-3", "google/pegasus-xsum", "facebook/bart-large-cnn", "google/pegasus-large"]


TEXT_NUM_TOPICS = 'Choose number of topics:'
TEXT_NER = "Named Entity Recognition"
TEXT_EMBEDDING = "Choose embedding type:"
TEXT_REV_EXPANDER = "Show data:"
TEXT_REV_SELECTOR = "Select episode review: "
TEXT_ENTRY_SELECTOR = "Select entry: "
TEXT_INPUT_SUMMARY = "Try writing your own summary and we'll compare it to our output:"


TITLE = "Game Of Thrones Summary Generator"
TITLE_EDA = "1. Exploratory Data Analysis"
TITLE_FILTER = "2. Filter text"
TITLE_SUMMARY = "3. Summary generator"
TITLE_METRICS = "4. Summary evaluation"
SPACY_LANG_MODEL = 'en_core_web_lg'
