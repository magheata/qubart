import re

import streamlit as st
from constants import *
import utils

st.set_option('deprecation.showPyplotGlobalUse', False)

# region SIDEBAR
review = st.sidebar.selectbox("Select episode review: ", [f"Episode {i + 1}" for i in range(0, 73)], key="season")
model_selected = st.sidebar.selectbox("Select desired model", MODELS)
dim_red = st.sidebar.selectbox(
 'Select dimension reduction method',
 ('PCA','TSNE'))
dimension = st.sidebar.radio(
     "Select the dimension of the visualization",
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
#endregion

# region SELECTED REVIEW
utils.title("Game Of Thrones Summary Generator", 60)
corpus_clean = utils.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS_CLEAN}")['train']
corpus = utils.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS}")['train']

selected_ep = int(re.search(r'\d+', review).group()) - 1
utils.title(f"Season {corpus_clean['Season'][selected_ep]},"
            f" Episode {corpus_clean['Episode'][selected_ep]}."
            f" {corpus_clean['Episode title'][selected_ep]}", 30, STREAMLIT_COLOR_TITLE)
episode_got_clean = corpus_clean['Episode recap'][int(re.search(r'\d+', review).group()) - 1]
episode_got = corpus['Episode recap'][int(re.search(r'\d+', review).group()) - 1]
sentences = episode_got_clean.split("_")
expander = st.expander("Show Episode Review")
expander.write(episode_got)

embedded_sentences = utils.get_sentence_embeddings(sentences)
#endregion

#region EDA
utils.title("Exploratory Data Analysis", 40, STREAMLIT_COLOR_TITLE)

if dimension == '2D':
    st.header('2D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    utils.display_scatterplot_2D(embedded_sentences, sentences, dim_red, perplexity, learning_rate, iteration)
else:
    st.header('3D Visualization')
    st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    utils.display_scatterplot_3D(embedded_sentences, sentences,dim_red, perplexity, learning_rate, iteration)
# endregion

# region FILTER TEXT
utils.title("Filter text", 40, STREAMLIT_COLOR_TITLE)

filter_by = st.text_input("Filter with", "Bran")

similarity_threshold = st.slider('Similarity threshold', 0.0, 1.0, 0.3)

similar_sentences = utils.get_similar_sentences(utils.get_sentence_embeddings(filter_by), embedded_sentences,
                                          similarity_threshold=similarity_threshold)
filtered_text = ""
for idx in similar_sentences:
    filtered_text = filtered_text + " " + sentences[idx]

utils.title('Filtered text', 30, STREAMLIT_COLOR_TITLE)
filtered_text
# endregion

# region SUMMARY
utils.title("Summary", 40, STREAMLIT_COLOR_TITLE)
summary = utils.get_summary(filtered_text, model_selected)
summary[0]
# endregion