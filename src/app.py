"""

"""
from constants import *
from streamlit import components
import re
import spacy
import spacy_streamlit
import streamlit as st
import utils as ut
from rouge import Rouge

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_option('deprecation.showPyplotGlobalUse', False)

review = st.sidebar.selectbox(TEXT_REV_SELECTOR, [f"Episode {i + 1}" for i in range(0, 73)], key="season")

# region SELECTED REVIEW
ut.title(TITLE, size=60, color=STREAMLIT_COLOR_TITLE)
# Data used in the filtering and summarization operations.
corpus_clean = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS_CLEAN}")['train']
# Data used for visualization purposes.
corpus = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS}")['train']
# Parse the selected review to the actual index in the corpus
selected_ep = int(re.search(r'\d+', review).group()) - 1
# Print the selected episode's information (season, episode in season, episode name)
ut.title(f"Season {corpus_clean[REV_SEASON][selected_ep]},"
            f" Episode {corpus_clean[REV_EPISODE][selected_ep]}."
            f" {corpus_clean[REV_TITLE][selected_ep]}", 30, STREAMLIT_COLOR_SUBTITLE)
# Review of the selected episode, used in the filtering and summarization operations.
episode_got_clean = corpus_clean[REV_REVIEW][int(re.search(r'\d+', review).group()) - 1]
# Review of the selected episode, used for visualization purposes.
episode_got = corpus[REV_REVIEW][int(re.search(r'\d+', review).group()) - 1]
# Get a list of all the sentences from the review, appending a "." at the end of each one.
sentences = episode_got_clean.split(SEPARATOR)
sentences = [sentence + "." for sentence in sentences]
# Show the episode review
expander = st.expander(TEXT_REV_EXPANDER)
expander.write(episode_got)
# Create the embeddings of the sentences
embedded_sentences = ut.get_sentence_embeddings(sentences)
#endregion

#region EDA
ut.title(TITLE_EDA, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("EDA options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
dim_red = st.sidebar.selectbox(TEXT_DIM_REDUCTION, OPTIONS_DIM_REDUCTION)

dim_reductor = ut.DimReductor(embedded_sentences)

topics = st.sidebar.slider(TEXT_NUM_TOPICS, 1, 10, 5)

show_ner = st.sidebar.radio(TEXT_NER, OPTIONS_NER)


st.header('2D Visualization')
st.write('For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences, sentences_text=sentences)

# Creating a spaCy object
nlp = spacy.load(SPACY_LANG_MODEL)
if show_ner == "On":
    doc = nlp(episode_got)
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)

# Creating a vectorizer
components.v1.html(ut.get_LDA_visualizer(episode_got_clean, topics), width=1300, height=875, scrolling=True)
# endregion

# region FILTER TEXT
ut.title(TITLE_FILTER, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Filtering options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
filter_by = st.sidebar.text_input("Filter with:", "Bran")
similarity_threshold = st.sidebar.slider('Choose similarity threshold:', 0.0, 1.0, 0.3)
if similarity_threshold > 0.3:
    st.sidebar.warning("Filtered text might not be what you expected, consider decreasing threshold. ")
similar_sentences = ut.get_similar_sentences(ut.get_sentence_embeddings(filter_by), embedded_sentences,
                                          similarity_threshold=similarity_threshold)
filtered_text = ""
filtered_text_clean = ""
for idx in similar_sentences:
    filtered_text_clean = filtered_text_clean + " " + sentences[idx]
    filtered_text = filtered_text + SEPARATOR + sentences[idx]

if filtered_text_clean == "":
    st.warning('Could not filter text with given input.')
else:
    filtered_text_clean
    if show_ner == "On":
        doc = nlp(filtered_text_clean)
        spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="filtered_text")
    components.v1.html(ut.get_LDA_visualizer(filtered_text_clean, topics), width=1300, height=875, scrolling=True)
    embedded_sentences_filtered = ut.get_sentence_embeddings(filtered_text.split(SEPARATOR)[1:])

    ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.5,
                              embeddings_filtered=embedded_sentences_filtered, sentences_filtered=filtered_text.split(SEPARATOR)[1:],filtered_marker_opacity=0.8)

# endregion

# region SUMMARY
ut.title(TITLE_SUMMARY, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Summary options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
model_selected = st.sidebar.selectbox("Select desired model:", MODELS)
if filtered_text == "":
    st.warning('Generating summary with whole text, filter could not be applied.')
    summary = ut.get_summary(episode_got_clean, model_selected)
else:
    summary = ut.get_summary(filtered_text, model_selected)
summary[0]
if show_ner == "On":
    doc = nlp(summary[0])
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="summary")
components.v1.html(ut.get_LDA_visualizer(summary[0], topics), width=1300, height=875, scrolling=True)
sentences_summary = summary[0].split('.')[:-1]
sentences_summary = [sentence + "." for sentence in sentences_summary]
embedded_sentences_summary = ut.get_sentence_embeddings(sentences_summary)

if filtered_text_clean == "":
    ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                              embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.5,
                              embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary, summary_marker_opacity=0.8)
else:
    ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                              embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.3,
                              embeddings_filtered=embedded_sentences_filtered, sentences_filtered=filtered_text.split(SEPARATOR)[1:], filtered_marker_opacity=0.5,
                              embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary, summary_marker_opacity=0.8)

# https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460
ut.title(TITLE_METRICS, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
rouge = Rouge()
scores_orig = rouge.get_scores(summary[0], episode_got_clean)
table = None
if filtered_text != "":
    scores_filtered = rouge.get_scores(summary[0], filtered_text)
    table = ut.get_df_rouge_scores([scores_orig[0], scores_filtered[0]], ["rouge-1", "rouge-2", "rouge-l"],
                                   index=["Original text (R1)", "Filtered text (R1)",
                                          "Original text (R2)", "Filtered text (R2)",
                                          "Original text (RL)", "Filtered text (RL)"])
else:
    table = ut.get_df_rouge_scores([scores_orig[0]], ["rouge-1", "rouge-2", "rouge-l"],
                                   index=["Original text (R1)", "Original text (R2)", "Original text (RL)"])

st.table(table)

input_summary = st.text_input("Try writing your own summary and we'll compare it to our output:")
if input_summary != "":
    scores = rouge.get_scores(summary[0], input_summary)
    table = ut.get_df_rouge_scores([scores[0]], ["rouge-1", "rouge-2", "rouge-l"],
                                   index=["Input text (R1)", "Input text (R2)", "Input text (RL)"])
    st.table(table)
# endregion