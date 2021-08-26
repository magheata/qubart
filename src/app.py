from constants import *
from streamlit import components
import re
import spacy
import spacy_streamlit
import streamlit as st
import utils as ut


st.set_option('deprecation.showPyplotGlobalUse', False)

# region SIDEBAR
review = st.sidebar.selectbox("Select episode review: ", [f"Episode {i + 1}" for i in range(0, 73)], key="season")


#endregion

# region SELECTED REVIEW
ut.title("Game Of Thrones Summary Generator", size=60, color=STREAMLIT_COLOR_TITLE)
corpus_clean = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS_CLEAN}")['train']
corpus = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS}")['train']

selected_ep = int(re.search(r'\d+', review).group()) - 1
ut.title(f"Season {corpus_clean['Season'][selected_ep]},"
            f" Episode {corpus_clean['Episode'][selected_ep]}."
            f" {corpus_clean['Episode title'][selected_ep]}", 30, STREAMLIT_COLOR_SUBTITLE)
episode_got_clean = corpus_clean['Episode recap'][int(re.search(r'\d+', review).group()) - 1]
episode_got = corpus['Episode recap'][int(re.search(r'\d+', review).group()) - 1]
sentences = episode_got_clean.split("_")
expander = st.expander("Show Episode Review")
expander.write(episode_got)

embedded_sentences = ut.get_sentence_embeddings(sentences)
#endregion

#region EDA
ut.title("1. Exploratory Data Analysis", 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("EDA options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
dim_red = st.sidebar.selectbox(
 'Select dimension reduction method:',
 ('PCA','TSNE'))
dimension = st.sidebar.radio(
     "Select the dimension of the visualization:",
     ('2D', '3D'))

if dim_red == 'TSNE':
    perplexity = st.sidebar.slider(
        'Adjust the perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity',
        5, 50, (30))

    learning_rate = st.sidebar.slider('Adjust the learning rate',
                                      10, 1000, (200))

    iteration = st.sidebar.slider('Adjust the number of iteration',
                                  250, 100000, (1000))
else:
    perplexity = 0
    learning_rate = 0
    iteration = 0


if dimension == '2D':
    st.header('2D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    ut.display_scatterplot_2D(embedded_sentences, sentences, dim_red, perplexity, learning_rate, iteration)
else:
    st.header('3D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    ut.display_scatterplot_3D(embedded_sentences, sentences,dim_red, perplexity, learning_rate, iteration)

# Creating a spaCy object
nlp = spacy.load('en_core_web_lg')
doc = nlp(episode_got)
spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)

# Creating a vectorizer
NUM_TOPICS = 10
components.v1.html(ut.get_LDA_visualizer(episode_got_clean, NUM_TOPICS), width=1300, height=875, scrolling=True)
# endregion

# region FILTER TEXT
ut.title("2. Filter text", 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Filtering options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
filter_by = st.sidebar.text_input("Filter with:", "Bran")
similarity_threshold = st.sidebar.slider('Choose similarity threshold:', 0.0, 1.0, 0.3)
similar_sentences = ut.get_similar_sentences(ut.get_sentence_embeddings(filter_by), embedded_sentences,
                                          similarity_threshold=similarity_threshold)
filtered_text = ""

for idx in similar_sentences:
    filtered_text = filtered_text + " " + sentences[idx]

if filtered_text == "":
    st.warning('Could not filter text with given input.')
else:
    filtered_text
    doc = nlp(filtered_text)
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="filtered_text")

# endregion

# region SUMMARY
ut.title("3. Summary generator", 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Summary options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
model_selected = st.sidebar.selectbox("Select desired model:", MODELS)
if filtered_text == "":
    st.warning('Generating summary with whole text, filter could not be applied.')
    summary = ut.get_summary(episode_got_clean, model_selected)
else:
    summary = ut.get_summary(filtered_text, model_selected)
summary[0]
doc = nlp(summary[0])
spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="summary")
components.v1.html(ut.get_LDA_visualizer(summary[0], NUM_TOPICS), width=1300, height=875, scrolling=True)
# endregion
page_bg_img = '''
<style>
body {
background-image: url("https://i.imgur.com/KeyMrcF.jpeg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)