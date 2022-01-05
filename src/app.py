"""
@author: Miruna Andreea Gheata
"""

import io
import os
import pandas as pd
import re
import spacy
import streamlit as st
import time
import utils as ut

from constants import *
from datasets import Dataset
from streamlit import components

# region STREAMLIT APPLICATION PARAMETERS AND SETTINGS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_option('deprecation.showPyplotGlobalUse', False)


# endregion

# region TEXT SELECTION
def use_existing_corpus():
    """
    Method used when the user chooses to use the Game of Thrones dataset as corpus.
    :return: data containing the chosen episode.
    """
    # Show the title of the Streamnlit page.
    ut.title(TITLE, size=60, color=STREAMLIT_COLOR_TITLE)
    # Load the Game of Thrones corpus.
    corpus = ut.load_corpus(f"corpus/{CORPUS_GOT_REVIEWS}")['train']
    # Prompt the user if they want to select the whole show or just an episode.
    whole_show = st.sidebar.radio("Select episode or use whole dataset?", ["Select episode", "Use whole dataset"])
    data = ""
    # If the user has chosen to use only an episode as input text
    if whole_show == "Select episode":
        # Select the chosen episode.
        review = st.sidebar.selectbox(TEXT_REV_SELECTOR, [f"Episode {i + 1}" for i in range(0, corpus.num_rows)],
                                      key="season")
        # Parse the selected review to the actual index in the corpus.
        selected_ep = int(re.search(r'\d+', review).group()) - 1
        # Print the selected episode's information (season, episode in season, episode name).
        ut.title(f"Season {corpus[REV_SEASON][selected_ep]},"
                 f" Episode {corpus[REV_EPISODE][selected_ep]}."
                 f" {corpus[REV_TITLE][selected_ep]}", 30, STREAMLIT_COLOR_SUBTITLE)
        # Review of the selected episode, used for visualization purposes.
        data = corpus[REV_REVIEW][selected_ep]
    else:
        # Concatenate the text of all episodes to a single text.
        for episode in range(0, corpus.num_rows):
            data = data + " " + corpus[REV_REVIEW][episode]
    # Return the text of the selected option.
    return data


def use_new_csv(input):
    """
    Method used to select the input data from a CSV local file.
    :param input: name of the CSV file.
    :return:
    """
    data = io.BytesIO(input.getbuffer())
    file_container = st.expander(f"File name: {input.name}")
    # Read the CSV file.
    csv_file = pd.read_csv(data)
    file_container.write(csv_file)
    corpus = Dataset.from_pandas(csv_file)
    # Show the different columns found in the CSV file and wait for user to choose the column that contains the desired
    # texts to summarize.
    data_column = st.selectbox("Choose column containing the data:", corpus.features)

    # Check if selected column contains text; if not, show error message and stop the application. Wait for correct
    # column type to be chosen.
    if type(corpus[data_column][0]) is not str:
        st.error("Selected column not string, please select another one. ")
        st.stop()

    # If column has more than one row, create select box with the different entries available for the user to choose
    # from.
    if corpus.num_rows > 1:
        # Create selectbox with different entries (rows).
        entry_num = st.sidebar.selectbox(TEXT_ENTRY_SELECTOR, [f"Entry {i + 1}" for i in range(0, corpus.num_rows)],
                                         key="entry")
        # Select the chosen entry from the options and get the text.
        selected_entry = int(re.search(r'\d+', entry_num).group()) - 1
        entry = corpus[data_column][selected_entry]
    else:
        # If only one row in the CSV file, return the text of this row.
        entry = corpus[data_column][0]
    return entry


# Make user choose type of data to be used.
used_corpus = st.sidebar.radio("Use exising corpus or load new data?", ["Existing corpus", "Load new data"])

# If user want new data
if used_corpus == "Load new data":
    # User can choose from uploading a text file
    text_uploader = st.file_uploader("Choose data you want to use:", type=AVAILABLE_TEXT_FORMATS)
    # Or typing a text.
    text_input = st.text_input("Or write your own data: ")
    # If user has selected a text file and written a text, return error message
    if text_uploader and text_input:
        st.error("Choose either uploading a file or your written input. ")
        st.stop()
    # If user has uploaded a text file
    elif text_uploader:
        success_msg = st.success("Data uploaded correctly!")
        time.sleep(3)
        success_msg.empty()
        # If the file has the correct format, load the data.
        if text_uploader.type == "text/csv":
            entry = use_new_csv(text_uploader)
    # If user has typed a text
    elif text_input:
        success_msg = st.success("Data uploaded correctly!")
        time.sleep(3)
        success_msg.empty()
        # The data to be used will be the text provided.
        entry = text_input
    # If error, notify user and stop application.
    else:
        st.error("Could not load data, please try again.")
        st.stop()
# If user wants to use existing data, load Game of Thrones corpus.
else:
    entry = use_existing_corpus()

# Get a list of all the sentences from the review, appending a "." at the end of each one.
sentences = re.split(REGEX_EOS, entry)
sentences = [sentence + "." for sentence in sentences]
# Show the episode review.
expander = st.expander(TEXT_REV_EXPANDER)
expander.write(entry)
# Create the embeddings of the sentences.
type_embedding = st.sidebar.radio(TEXT_EMBEDDING, OPTIONS_EMBEDDING)
embedded_sentences = ut.get_sentence_embeddings(sentences)
# Create instance of the dimensionality reductor.
dim_reductor = ut.DimReductor(embedded_sentences)

# If user chose doc2vec, embed the whole entry text.
if type_embedding == "doc2vec":
    embedded_document = ut.get_sentence_embeddings(entry)
# endregion

# region EXPLORATORY DATA ANALYSIS
# Show titles for this section
ut.title(TITLE_EDA, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("EDA options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)

# The user can choose how many topics to look for in the text.
topics = st.sidebar.slider(TEXT_NUM_TOPICS, 1, 10, 5)

# If sent2vec chosen, show the scatterplot of the sentence embeddings.
if type_embedding == "sent2vec":
    st.header('2D Visualization')
    st.write(
        'For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization.')
    ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences, sentences_text=sentences)


# Computing and showing the different topics found in the entry text.
components.v1.html(ut.get_LDA_visualizer(entry, topics), width=1300, height=875, scrolling=True)
# endregion

# region TEXT FILTERING
# Show titles for this section.
ut.title(TITLE_FILTER, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Filtering options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
# Select the filter used in the filtering of the text. Default value is "Bran".
filter_by = st.sidebar.text_input("Filter with:", "Bran")
# Select the similarity threshold that will be used to filter the sentences. The higher the threshold value, the more
# similar is the text to the filter.
similarity_threshold = st.sidebar.slider('Choose similarity threshold:', 0.0, 1.0, 0.3)
# Obtain the similar sentences.
similar_sentences = ut.get_similar_sentences(ut.get_sentence_embeddings(filter_by), embedded_sentences,
                                             similarity_threshold=similarity_threshold)
# Filtered text used in the application.
filtered_text = ""
# Filtered text used to show to the user.
filtered_text_clean = ""
# Concatenate all similar sentences found.
for idx in similar_sentences:
    filtered_text_clean = filtered_text_clean + " " + sentences[idx]
    filtered_text = filtered_text + SEPARATOR + sentences[idx]

# If filtered text is empty, notify user and prompt for a similarity threshold change.
if filtered_text_clean == "":
    st.warning('Could not filter text with given input.')
    st.sidebar.warning("Consider decreasing threshold to ensure filtering is possible.")
# If filtered text is not empty
else:
    # Show user the filtered text obtained after applying the filter.
    filtered_text_clean

    # Computing and showing the different topics found in the filtered text.
    components.v1.html(ut.get_LDA_visualizer(filtered_text_clean, topics), width=1300, height=875, scrolling=True)
    # If sentence embedding selected, compute and show the sentence embeddings.
    if type_embedding == "sent2vec":
        embedded_sentences_filtered = ut.get_sentence_embeddings(filtered_text.split(SEPARATOR)[1:])
        ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_sentences,
                                  sentences_text=sentences, marker_opacity=0.5,
                                  embeddings_filtered=embedded_sentences_filtered,
                                  sentences_filtered=filtered_text.split(SEPARATOR)[1:], filtered_marker_opacity=0.8)
    # If document embedding selected, compute and show the document embedding.
    else:
        embedded_document_filtered = ut.get_sentence_embeddings(filtered_text)
        ut.display_scatterplot_2D(dim_reductor=dim_reductor, embeddings_text=embedded_document, sentences_text=entry,
                                  marker_opacity=0.5,
                                  embeddings_filtered=embedded_document_filtered,
                                  sentences_filtered=filtered_text.replace("_", "."), filtered_marker_opacity=0.8)
# endregion

# region SUMMARY GENERATION
# Show titles for this section.
ut.title(TITLE_SUMMARY, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
ut.title("Summary options", size=20, color=STREAMLIT_COLOR_TITLE, sidebar=True)
# Get the selected Transformer model by the user.
model_selected = st.sidebar.selectbox("Select desired model:", MODELS)
# If no filtered text, the summary will be created with the original text (entry).
if filtered_text == "":
    st.warning('Generating summary with whole text, filter could not be applied.')
    summary = ut.get_summary(entry, model_selected)
# If filtered text available, the summary will be created with it.
else:
    summary = ut.get_summary(filtered_text, model_selected)
# Show summary to the user.
summary[0]

# Computing and showing the different topics found in the generated summary.
components.v1.html(ut.get_LDA_visualizer(summary[0], topics), width=1300, height=875, scrolling=True)

# Split the summary into sentences.
sentences_summary = summary[0].split('.')[:-1]
sentences_summary = [sentence + "." for sentence in sentences_summary]
# Create sentence and document embeddings for the generated summary.
embedded_sentences_summary = ut.get_sentence_embeddings(sentences_summary)
embedded_document_summary = ut.get_sentence_embeddings(summary[0])

# If no filtered text available
if filtered_text_clean == "":
    # Show sentence embeddings plot
    if type_embedding == "sent2vec":
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.5,
                                  embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary,
                                  summary_marker_opacity=0.8)
    # Show document embeddings plot
    else:
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_document, sentences_text=entry, marker_opacity=0.5,
                                  embeddings_summary=embedded_document_summary, sentences_summary=summary[0],
                                  summary_marker_opacity=0.8)
# If filtered text available
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
# endregion

# region SUMMARY EVALUATION
# Reference-> https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460
# Show titles for this section.
ut.title(TITLE_METRICS, 40, STREAMLIT_COLOR_SUBTITLE, text_align="left")
# Get ROUGE scores using the original text (entry) as reference.
scores_orig = ut.get_rouge(summary[0], entry)

# If filtered text available, compute the ROUGE scores using the filtered text as reference.
if filtered_text != "":
    scores_filtered = ut.get_rouge(summary[0], filtered_text)
    # Choose ROUGE metric used to compare to.
    ref_text = st.selectbox("Select reference text:", TEXT_REF_OPTIONS_FILTER, key="filtered_text")
else:
    # Choose ROUGE metric used to compare to.
    ref_text = st.selectbox("Select reference text:", TEXT_REF_OPTIONS_NO_FILTER, key="no_filtered_text")

# Create Streamlit columns to show the comparison between the ROUGE metrics.
col1, col2, col3 = st.columns(3)

# Get the comparison between the using the original text as reference and using another text as reference.
ut.get_rouge_metric_comparison(col1, col2, col3,
                               ref_text,
                               hypothesys=summary[0],
                               rouge_score=scores_orig[0],
                               filtered_text=filtered_text if filtered_text != "" else None,
                               baseline=entry[:LEAD_N])
# Show plot with the comparison of the scores obtained when using the different available reference texts.
st.write(ut.get_plot_rouge(orig_score=ut.get_rouge(summary[0], entry)[0]["rouge-l"]["f"],
                           baseline_score=ut.get_rouge(summary[0], entry[:LEAD_N])[0]["rouge-l"]["f"],
                           filtered_score=ut.get_rouge(summary[0], filtered_text)[0]["rouge-l"][
                               "f"] if filtered_text else None))
# The user can also input a summary that can be compared to the generated summary.
input_summary = st.text_input(TEXT_INPUT_SUMMARY)
if input_summary != "":
    # If sentence embeddings selected, show the sentence embeddings of the user's input summary.
    if type_embedding == "sent2vec":
        embedded_sentences_input = ut.get_sentence_embeddings(input_summary.split(".")[:-1])
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_sentences, sentences_text=sentences, marker_opacity=0.3,
                                  embeddings_summary=embedded_sentences_summary, sentences_summary=sentences_summary,
                                  summary_marker_opacity=0.8,
                                  embeddings_input=embedded_sentences_input, sentences_input=input_summary.split("."),
                                  input_marker_opacity=0.8)
    # If document embeddings selected, show the document embeddings of the user's input summary.
    else:
        embedded_document_input = ut.get_sentence_embeddings(input_summary)
        ut.display_scatterplot_2D(dim_reductor=dim_reductor,
                                  embeddings_text=embedded_document, sentences_text=entry, marker_opacity=0.3,
                                  embeddings_summary=embedded_document_summary, sentences_summary=summary[0],
                                  summary_marker_opacity=0.8,
                                  embeddings_input=embedded_document_input, sentences_input=input_summary,
                                  input_marker_opacity=0.8)
    # Show the ROUGE scores comparison.
    st.write(ut.get_plot_rouge(orig_score=ut.get_rouge(summary[0], entry)[0]["rouge-l"]["f"],
                               baseline_score=ut.get_rouge(summary[0], entry[:LEAD_N])[0]["rouge-l"]["f"],
                               filtered_score=ut.get_rouge(summary[0], filtered_text)[0]["rouge-l"][
                                   "f"] if filtered_text else None,
                               user_score=ut.get_rouge(summary[0], input_summary)[0]["rouge-l"][
                                   "f"] if input_summary else None))

    # Compute the ROUGE score using the generated summary and user summary as reference.
    scores = ut.get_rouge(summary[0], input_summary)

    comparison_scores = {}

    if filtered_text != "":
        ref_options = TEXT_REF_OPTIONS_USER_FILTER
        comparison_scores = scores_filtered[0]
    else:
        ref_options = TEXT_REF_OPTIONS_USER_NO_FILTER
        comparison_scores = scores_orig[0]

    ref_text = st.selectbox("Select reference text:", ref_options, key="input_text")

    col1, col2, col3 = st.columns(3)
    ut.get_rouge_metric_comparison(col1, col2, col3,
                                   hypothesys=summary[0],
                                   ref_text=ref_text,
                                   rouge_score=scores[0], baseline=entry[:LEAD_N],
                                   entry=entry,
                                   filtered_text=filtered_text if filtered_text != "" else None)

# endregion
