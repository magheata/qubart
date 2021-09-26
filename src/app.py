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
import time
import io
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import umap
import umap.plot

def use_existing_corpus():
    ut.title(TITLE, size=60, color=STREAMLIT_COLOR_TITLE)
    corpus = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS}")['train']
    whole_show = st.sidebar.radio("Select episode or use whole dataset?", ["Select episode", "Use whole dataset"])
    data = ""
    if whole_show == "Select episode":
        review = st.sidebar.selectbox(TEXT_REV_SELECTOR, [f"Episode {i + 1}" for i in range(0, corpus.num_rows)],
                                      key="season")
        # Parse the selected review to the actual index in the corpus
        selected_ep = int(re.search(r'\d+', review).group()) - 1
        # Print the selected episode's information (season, episode in season, episode name)
        ut.title(f"Season {corpus[REV_SEASON][selected_ep]},"
                 f" Episode {corpus[REV_EPISODE][selected_ep]}."
                 f" {corpus[REV_TITLE][selected_ep]}", 30, STREAMLIT_COLOR_SUBTITLE)
        # Review of the selected episode, used for visualization purposes.
        data = corpus[REV_REVIEW][selected_ep]
    else:
        for episode in range(0, corpus.num_rows):
            data = data + " " + corpus[REV_REVIEW][episode]
    print(data)
    return data


def use_new_csv(input):
    data = io.BytesIO(input.getbuffer())
    file_container = st.expander(f"File name: {input.name}")
    csv_file = pd.read_csv(data)
    file_container.write(csv_file)
    corpus = Dataset.from_pandas(csv_file)
    data_column = st.selectbox("Choose column containing the data:", corpus.features)

    if type(corpus[data_column][0]) is not str:
        st.error("Selected column not string, please select another one. ")
        st.stop()

    if corpus.num_rows > 1:
        entry_num = st.sidebar.selectbox(TEXT_ENTRY_SELECTOR, [f"Entry {i + 1}" for i in range(0, corpus.num_rows)],
                                         key="entry")
        selected_entry = int(re.search(r'\d+', entry_num).group()) - 1
        entry = corpus[data_column][selected_entry]
        print(corpus.num_rows)
    else:
        entry = corpus[data_column][0]
    print("State", st.session_state)

    return entry


os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_option('deprecation.showPyplotGlobalUse', False)

used_corpus = st.sidebar.radio("Use exising corpus or load new data?", ["Existing corpus", "Load new data"])

if used_corpus == "Load new data":
    text_uploader = st.file_uploader("Choose data you want to use:", type=["csv", "txt", "docx", "xlsx"])
    text_input = st.text_input("Or write your own data: ")
    if text_uploader and text_input:
        st.error("Choose either uploading a file or your written input. ")
        st.stop()
    elif text_uploader:
        success_msg = st.success("Data uploaded correctly!")
        time.sleep(3)
        success_msg.empty()
        if text_uploader.type == "text/csv":
            entry = use_new_csv(text_uploader)
    elif text_input:
        success_msg = st.success("Data uploaded correctly!")
        time.sleep(3)
        success_msg.empty()
        entry = text_input
    else:
        st.error("Could not load data, please try again.")
        st.stop()
else:
    print("State", st.session_state)
    entry = use_existing_corpus()

# Get a list of all the sentences from the review, appending a "." at the end of each one.
sentences = re.split(REGEX_EOS, entry)
sentences = [sentence + "." for sentence in sentences]
# Show the episode review
expander = st.expander(TEXT_REV_EXPANDER)
expander.write(entry)
# Create the embeddings of the sentences
type_embedding = st.sidebar.radio(TEXT_EMBEDDING, OPTIONS_EMBEDDING)
embedded_sentences = ut.get_sentence_embeddings(sentences)
dim_reductor = ut.DimReductor(embedded_sentences)

if type_embedding == "doc2vec":
    embedded_document = ut.get_sentence_embeddings(entry)
# endregion

# region EDA
ut.title(TITLE_EDA, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("EDA options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)

topics = st.sidebar.slider(TEXT_NUM_TOPICS, 1, 10, 5)

show_ner = st.sidebar.radio(TEXT_NER, OPTIONS_NER)

if type_embedding == "sent2vec":
    st.header('2D Visualization')
    st.write(
        'For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences, sentences_text=sentences)

# Creating a spaCy object
nlp = spacy.load(SPACY_LANG_MODEL)
if show_ner == "On":
    doc = nlp(entry)
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)

# Creating a vectorizer
components.v1.html(ut.get_LDA_visualizer(entry, topics), width=1300, height=875, scrolling=True)
# endregion

# region FILTER TEXT
ut.title(TITLE_FILTER, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Filtering options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
filter_by = st.sidebar.text_input("Filter with:", "Bran")
similarity_threshold = st.sidebar.slider('Choose similarity threshold:', 0.0, 1.0, 0.3)
similar_sentences = ut.get_similar_sentences(ut.get_sentence_embeddings(filter_by), embedded_sentences,
                                             similarity_threshold=similarity_threshold)
filtered_text = ""
filtered_text_clean = ""
for idx in similar_sentences:
    filtered_text_clean = filtered_text_clean + " " + sentences[idx]
    filtered_text = filtered_text + SEPARATOR + sentences[idx]

if filtered_text_clean == "":
    st.warning('Could not filter text with given input.')
    st.sidebar.warning("Consider decreasing threshold to ensure filtering is possible.")
else:
    filtered_text_clean
    if show_ner == "On":
        doc = nlp(filtered_text_clean)
        spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="filtered_text")
    components.v1.html(ut.get_LDA_visualizer(filtered_text_clean, topics), width=1300, height=875, scrolling=True)
    if type_embedding == "sent2vec":
        embedded_sentences_filtered = ut.get_sentence_embeddings(filtered_text.split(SEPARATOR)[1:])
        ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences,
                                  sentences_text=sentences, marker_opacity=0.5,
                                  embeddings_filtered=embedded_sentences_filtered,
                                  sentences_filtered=filtered_text.split(SEPARATOR)[1:], filtered_marker_opacity=0.8)
    else:
        embedded_document_filtered = ut.get_sentence_embeddings(filtered_text)
        ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_document, sentences_text=entry,
                                  marker_opacity=0.5,
                                  embeddings_filtered=embedded_document_filtered,
                                  sentences_filtered=filtered_text.replace("_", "."), filtered_marker_opacity=0.8)

# endregion

# region SUMMARY
ut.title(TITLE_SUMMARY, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Summary options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
model_selected = st.sidebar.selectbox("Select desired model:", MODELS)
if filtered_text == "":
    st.warning('Generating summary with whole text, filter could not be applied.')
    summary = ut.get_summary(entry, model_selected)
else:
    print("Using filtered text")
    summary = ut.get_summary(filtered_text, model_selected)
summary[0]
if show_ner == "On":
    doc = nlp(summary[0])
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels, key="summary")
components.v1.html(ut.get_LDA_visualizer(summary[0], topics), width=1300, height=875, scrolling=True)
sentences_summary = summary[0].split('.')[:-1]
sentences_summary = [sentence + "." for sentence in sentences_summary]
embedded_sentences_summary = ut.get_sentence_embeddings(sentences_summary)
embedded_document_summary = ut.get_sentence_embeddings(summary[0])

if filtered_text_clean == "":
    if type_embedding == "sent2vec":
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.5,
                                  embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary,
                                  summary_marker_opacity=0.8)
    else:
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_document, sentences_text=entry, marker_opacity=0.5,
                                  embeddings_summary=embedded_document_summary, sentences_summary=summary[0],
                                  summary_marker_opacity=0.8)
else:
    if type_embedding == "sent2vec":
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.3,
                                  embeddings_filtered=embedded_sentences_filtered,
                                  sentences_filtered=filtered_text.split(SEPARATOR)[1:], filtered_marker_opacity=0.5,
                                  embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary,
                                  summary_marker_opacity=0.8)
    else:
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_document, sentences_text=entry, marker_opacity=0.3,
                                  embeddings_filtered=embedded_document_filtered,
                                  sentences_filtered=filtered_text.replace("_", "."), filtered_marker_opacity=0.5,
                                  embeddings_summary=embedded_document_summary, sentences_summary=summary[0],
                                  summary_marker_opacity=0.8)

# https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460
ut.title(TITLE_METRICS, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
rouge = Rouge()
scores_orig = rouge.get_scores(summary[0], entry)
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

input_summary = st.text_input(TEXT_INPUT_SUMMARY)
if input_summary != "":
    if type_embedding == "sent2vec":
        embedded_sentences_input = ut.get_sentence_embeddings(input_summary.split(".")[:-1])
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.3,
                                  embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary,
                                  summary_marker_opacity=0.8,
                                  embeddings_input=embedded_sentences_input, sentences_input=input_summary.split("."),
                                  input_marker_opacity=0.8)
    else:
        embedded_document_input = ut.get_sentence_embeddings(input_summary)
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_document, sentences_text=entry, marker_opacity=0.3,
                                  embeddings_summary=embedded_document_summary, sentences_summary=summary[0],
                                  summary_marker_opacity=0.8,
                                  embeddings_input=embedded_document_input, sentences_input=input_summary,
                                  input_marker_opacity=0.8)
    scores = rouge.get_scores(summary[0], input_summary)
    table = ut.get_df_rouge_scores([scores[0]], ["rouge-1", "rouge-2", "rouge-l"],
                                   index=["Input text (R1)", "Input text (R2)", "Input text (RL)"])
    st.table(table)
# endregion
