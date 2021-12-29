"""
@author: Miruna Andreea Gheata
"""

# region IMPORTS
import constants
import numpy as np
import plotly.graph_objs as go
import pyLDAvis.sklearn
import streamlit as st
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from rouge import Rouge

# endregion

# Instance of Rouge class that will be used to compute the ROUGE metric.
rouge = Rouge()


class DimReductor:
    def __init__(self, data):
        """
        Constructor of the DimReductor class. This class is used to reduce the dimensionality of the input
        data.
        :param data: element that will have its dimensionality reduced.
        """
        self.PCA = PCA(random_state=0).fit(data)


@st.cache(suppress_st_warning=True)
def load_corpus(data_files, ext_type='csv'):
    """
    Method used to load the corpus that will be used. The type of file can be chosen.
    :param data_files: path to the data file(s) that need to be loaded.
    :param ext_type: type of file. Default is 'csv'.
    :return: text in the given data file(s).
    """
    return load_dataset(ext_type, data_files=data_files)


def get_rouge(hypothesis, reference):
    """
    Method used to compute the ROUGE score between a hypothesis and a reference text. The higher the score, the more
    similar the hypothesis is to the reference text.
    :param hypothesis: text to be compared.
    :param reference: text used as correct answer.
    """
    return rouge.get_scores(hypothesis, reference)


# region TEXT PROCESSING
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_sentence_embeddings(input_data, model_name='sentence-transformers/paraphrase-mpnet-base-v2'):
    """
    Returns the embedded sentences of the input data provided.
    :param input_data: original text that we want to embed.
    :param model_name: Transformer model we will use for the tokenizer and the model.
    :return: list of embedded sentences.
    """

    # The tokenizer is used to prepare the input data for processing. The used tokenizer will be determined by the
    # model name provided.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The model is in charge of translating the tokens into the embedded sentences.
    model = AutoModel.from_pretrained(model_name)
    # We encode the input text with the tokenizer.
    encoded_input = tokenizer(input_data, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings.
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Return the embedded sentences.
    return sentence_embeddings.detach().numpy()


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    Method used to retain the mean information found in the sentence embeddings.
    :param model_output: sentence embeddings computed by the model.
    :param attention_mask: attention mask calculated by the model.
    :return: new sentence embeddings containing the mean information of the old ones.
    """
    # First element of model_output contains all token embeddings, or the last_hidden_state.
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache(suppress_st_warning=True)
def get_similar_sentences(sentence, other_sentences, similarity_threshold=0.7):
    """
    Method that applies the cosine similarity between a reference sentence and a list of other sentences and return those
    that are the most similar to the reference sentence.
    :param sentence: reference sentence.
    :param other_sentences: list of sentences to be compared to the reference sentences.
    :param similarity_threshold: threshold that will determine how similar the sentences need to be to the reference
    sentence.
    :return: list of sentences have a similarity to the reference sentence above the given threshold.
    """
    # Compute the cosine similarity of each of the sentences with respect to the reference sentence.
    sims = cosine_similarity(sentence, other_sentences)
    # Return those sentences that have a similarity score bigger than the given threshold.
    return np.where(sims.reshape(-1) >= similarity_threshold)[0]


@st.cache(suppress_st_warning=True)
def get_summary(input_text, model_name='sshleifer/distilbart-cnn-12-6'):
    """
    Method that creates the summary of a given input text using a Seq2Seq Summarization model. The model used will be provided by the
    user; if not, the default model is the BART model.
    :param input_text: text that we want to summarize.
    :param model_name: name of the Summarization model to be used.
    :return: summary of the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # encode input context.
    input_ids = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt").input_ids
    # generate summary.
    outputs = model.generate(input_ids=input_ids)
    # decode summary.
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# endregion

# region STREAMLIT REPRESENTATION METHODS
def title(text, size, color="white", text_align="center", h_type="h1", sidebar=False):
    """
    Method used to create the Streamlit tiles of the page.
    :param text: text to use as title.
    :param size: size of the text.
    :param color: color of the text.
    :param text_align: alignment of the text.
    :param h_type: type of header to use.
    :param sidebar: boolean that shows if the title will be in the sidebar.
    """
    # If the title is for the sidebar, use st.sidebar
    if sidebar:
        st.sidebar.markdown(f'<{h_type} style="font-weight:bolder;'
                            f'font-size:{size}px;'
                            f'color:{color};'
                            f'text-align:{text_align};">{text}</{h_type}>',
                            unsafe_allow_html=True)
    else:
        st.markdown(f'<{h_type} style="font-weight:bolder;'
                    f'font-size:{size}px;'
                    f'color:{color};'
                    f'text-align:{text_align};">{text}</{h_type}>',
                    unsafe_allow_html=True)


def header(text):
    """
    Method used to print a header into the Streamlit page.
    :param text: text used in the header.
    """
    st.markdown(f"<p style='color:white;'>{text}</p>", unsafe_allow_html=True)


def get_dim_reduction_data(dim_reductor, data, labels, text_legend,
                           dim_red='PCA', textposition="top center",
                           textfont_color="black", marker_color="black", marker_opacity=0.8):
    """
    Method used to get the data points that will be used in the visualization of the embeddings of the text.
    :param dim_reductor: object of the dimensionality reduction method used.
    :param data: data wanted to be reduced.
    :param labels: labels of the elements. Labels will be the sentences of the text.
    :param text_legend: shows the origin of the text, such as "Original Text", "Filtered text", etc.
    :param dim_red: method of dimensionality reduction.
    :param textposition: Position of the text.
    :param textfont_color: Color of the text.
    :param marker_color: Color of the plot.
    :param marker_opacity: Opacity of the marker in the plot.
    :return:
    """
    # Apply the dimensionality reduction to the input data and retain only two dimensions. This way, the data can be
    # plotted in a 2D plot.
    two_dim = dim_reductor.PCA.transform(data)[:, :2]
    count = 0

    trace_input = go.Scatter(
        x=two_dim[count:, 0],
        y=two_dim[count:, 1],
        text=labels[count:],
        name=text_legend,
        textposition=textposition,
        textfont={
            'size': 10,
            'color': textfont_color
        },
        mode='markers+text',
        marker={
            'size': 15,
            'opacity': marker_opacity,
            'color': marker_color
        }
    )
    return trace_input


def display_scatterplot_2D(dim_reductor, embeddings_text, sentences_text, marker_color="black", marker_opacity=0.8,
                           embeddings_summary=None, sentences_summary=None, summary_marker_color="red",
                           summary_marker_opacity=0.8,
                           embeddings_filtered=None, sentences_filtered=None, filtered_marker_color="blue",
                           filtered_marker_opacity=0.8,
                           embeddings_input=None, sentences_input=None, input_marker_color="purple",
                           input_marker_opacity=0.8
                           ):
    """
    Source -> https://github.com/marcellusruben/Word_Embedding_Visualization
    :param dim_reductor:
    :param embeddings_text:
    :param sentences_text:
    :param marker_color:
    :param marker_opacity:
    :param embeddings_summary:
    :param sentences_summary:
    :param summary_marker_color:
    :param summary_marker_opacity:
    :param embeddings_filtered:
    :param sentences_filtered:
    :param filtered_marker_color:
    :param filtered_marker_opacity:
    :param embeddings_input:
    :param sentences_input:
    :param input_marker_color:
    :param input_marker_opacity:
    :return:
    """

    data = []
    data.append(get_dim_reduction_data(dim_reductor=dim_reductor, data=embeddings_text, labels=sentences_text,
                                       text_legend='Original text', marker_color=marker_color,
                                       marker_opacity=marker_opacity))
    if embeddings_filtered is not None and sentences_filtered is not None:
        data.append(
            get_dim_reduction_data(dim_reductor=dim_reductor, data=embeddings_filtered, labels=sentences_filtered,
                                   text_legend='Filtered text', textposition="bottom center",
                                   textfont_color=filtered_marker_color,
                                   marker_color=filtered_marker_color, marker_opacity=filtered_marker_opacity))
    if embeddings_summary is not None and sentences_summary is not None:
        data.append(get_dim_reduction_data(dim_reductor=dim_reductor, data=embeddings_summary, labels=sentences_summary,
                                           text_legend='Summary', textposition="bottom center",
                                           textfont_color=summary_marker_color, marker_color=summary_marker_color,
                                           marker_opacity=summary_marker_opacity))
    if embeddings_input is not None and sentences_input is not None:
        data.append(get_dim_reduction_data(dim_reductor=dim_reductor, data=embeddings_input, labels=sentences_input,
                                           text_legend='Input summary', textposition="bottom center",
                                           textfont_color=input_marker_color, marker_color=input_marker_color,
                                           marker_opacity=input_marker_opacity))

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=15,
            font_family="Courier New"),
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=15,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)

    st.plotly_chart(plot_figure)


@st.cache(suppress_st_warning=True)
def spacy_tokenizer(sentence):
    """

    :param sentence:
    :return:
    """
    parser = English()
    mytokens = parser(sentence)
    mytokens = [word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in constants.STOPWORDS and word not in constants.PUNCTUATIONS]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def get_LDA_visualizer(data, topics):
    """
    Method used to represent the topics found through the application of Latent Dirichlet Allocation (LDA) to the
    data.
    :param data: text that will be analyzed to obtain the topics.
    :param topics: number of topics to find.
    :return: html object that contains the representation of the found topics.
    """
    # Obtain the tokens of the text.
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    tokens = spacy_tokenizer(data)
    data_vectorized = vectorizer.fit_transform([tokens])
    # Apply Latent Dirichlet Allocation Model.
    lda = LatentDirichletAllocation(n_components=topics, max_iter=10, learning_method='online', verbose=True)
    lda.fit_transform(data_vectorized)
    # Keywords for topics clustered by Latent Dirichlet Allocation
    dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds="pcoa")
    html = pyLDAvis.prepared_data_to_html(dash)
    return html


def get_rouge_metric_comparison(col1, col2, col3,
                                ref_text,
                                hypothesys,
                                rouge_score, baseline, entry=None,
                                filtered_text=None, round=3):
    """
    Method used to obtain a scatterplot containing the differente ROUGE scores obtained by comparing the resulting summary
    to different reference texts.
    :param col1: object used to represent the R1 Score.
    :param col2: object used to represent the R2 Score.
    :param col3: object used to represent the RL Score.
    :param ref_text: reference text used.
    :param hypothesys: ROUGE scores obtained.
    :param rouge_score: ROUGE score obtained by using the original text as reference.
    :param baseline: baseline text.
    :param entry: original text.
    :param filtered_text: filtered text.
    :param generated_summary: generated summary.
    :param round: rounding factor.
    """

    if entry and ref_text == "Original text":
        comparison_scores = get_rouge(hypothesys, entry)
    elif filtered_text and ref_text == "Filtered text":
        comparison_scores = get_rouge(hypothesys, filtered_text)
    else:
        comparison_scores = get_rouge(hypothesys, baseline)

    col1.metric(label="R-1", value=float.__round__(rouge_score["rouge-1"]["f"], round),
                delta=float.__round__(rouge_score["rouge-1"]["f"] - comparison_scores[0]["rouge-1"]["f"], round))
    col2.metric(label="R-2", value=float.__round__(rouge_score["rouge-2"]["f"], round),
                delta=float.__round__(rouge_score["rouge-2"]["f"] - comparison_scores[0]["rouge-2"]["f"], round))
    col3.metric(label="R-L", value=float.__round__(rouge_score["rouge-l"]["f"], round),
                delta=float.__round__(rouge_score["rouge-l"]["f"] - comparison_scores[0]["rouge-l"]["f"], round))


def get_plot_rouge(orig_score, baseline_score, filtered_score=None, user_score=None):
    """
    Method that created a scatter plot showing the different ROUGE scores obtained when comparing the generated summary
    to the different available reference texts.
    :param orig_score: Score obtained using the original text as reference.
    :param baseline_score: Score obtained using the baseline text as reference.
    :param filtered_score: Score obtained using the filtered text as reference.
    :param user_score: Score obtained using the user's summary as reference.
    :return: scatter plot figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x_values = []
    y_values = []
    plt.scatter(0, orig_score, alpha=.5, label="Original text")
    x_values.append(0)
    y_values.append(orig_score)

    plt.scatter(0, baseline_score, alpha=.5, label="Baseline")
    x_values.append(0)
    y_values.append(baseline_score)

    if filtered_score:
        plt.scatter(0, filtered_score, alpha=.5, label="Filtered text")
        x_values.append(0)
        y_values.append(filtered_score)
    if user_score:
        plt.scatter(0, user_score, alpha=.5, label="User summary")
        x_values.append(0)
        y_values.append(user_score)

    for x, y in zip(x_values, y_values):
        label = "{:.3f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     fontsize=7,
                     ha='center')  # horizontal alignment can be left, right or center

    plt.title("Comparison of ROUGE F-1 Scores depending on the Reference text used")
    plt.legend(title="Reference text used")
    plt.show()

    return fig
# endregion
