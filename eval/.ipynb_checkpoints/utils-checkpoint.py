import constants
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pyLDAvis.sklearn
import streamlit as st
import torch


from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel


class DimReductor():
    def __init__(self, data):
        self.PCA = PCA(random_state=0).fit(data)


def get_sentence_embeddings(input_data, model_name='sentence-transformers/paraphrase-mpnet-base-v2'):
    '''
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(input_data, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings.detach().numpy()


def load_corpus(data_files, ext_type='csv'):
    '''
    '''
    return load_dataset(ext_type, data_files=data_files)


def filter_corpus(corpus, element, filter_with):
    '''
    '''
    return corpus.filter(lambda filtered_corpus: filtered_corpus[element].startswith(filter_with))


def get_script_episode(corpus, season, episode):
    '''
    '''
    data_season = filter_corpus(corpus, 'Season', season)
    return filter_corpus(data_season, 'Episode', episode)


def get_full_text(input_text):
    return ''.join(input_text)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    '''
    '''
    token_embeddings = model_output[
        0]  # First element of model_output contains all token embeddings, or the last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_similar_sentences(sentence, other_sentences, similarity_threshold=0.7):
    '''
    '''
    sims = cosine_similarity(sentence, other_sentences)
    return np.where(sims.reshape(-1) >= similarity_threshold)[0]


def get_summary(input_text, model_name='sshleifer/distilbart-cnn-12-6'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # encode input context
    input_ids = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt").input_ids
    # generate summary
    outputs = model.generate(input_ids=input_ids)
    # decode summary
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def title(text, size, color="white", text_align="center", h_type="h1", sidebar=False):
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
    st.markdown(f"<p style='color:white;'>{text}</p>", unsafe_allow_html=True)


def get_dim_reduction_data(dim_reductor, data, labels, text_legend,
                           dim_red='PCA', textposition="top center",
                           textfont_color="black", marker_color="black", marker_opacity=0.8):
    if dim_red == 'PCA':
        two_dim = dim_reductor.PCA.transform(data)[:, :2]
    else:
        two_dim = dim_reductor.TSNE.transform(data)[:, :2]
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
    '''
    https://github.com/marcellusruben/Word_Embedding_Visualization
    '''

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


def spacy_tokenizer(sentence):
    parser = English()
    mytokens = parser(sentence)
    mytokens = [word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in constants.STOPWORDS and word not in constants.PUNCTUATIONS]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def get_LDA(data, topics):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    tokens = spacy_tokenizer(data)
    data_vectorized = vectorizer.fit_transform([tokens])
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=topics, max_iter=10, learning_method='online', verbose=True)
    data_lda = lda.fit_transform(data_vectorized)
    return lda, vectorizer, data_lda

def get_df_rouge_scores(scores, rouge_measures, columns, index, orig_length, summary_length, filtered, filtered_length = -1):
    rows = []
    
    for rouge_measure in rouge_measures:
        for score in scores:
            new_row = {'Original data length': orig_length, 
                       'Filtered data length': filtered_length, 
                       'Summary length': summary_length,
                       'Precision': score[rouge_measure]["r"],
                       'Recall': score[rouge_measure]["p"],
                       'F-Score': score[rouge_measure]["f"],
                       'Filtered': filtered}
            rows.append(new_row)
    dataframe = pd.DataFrame(rows, columns=columns, index=index)
    return dataframe