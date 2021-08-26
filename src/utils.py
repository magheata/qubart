import constants
import numpy as np
import plotly.graph_objs as go
import pyLDAvis.sklearn
import streamlit as st
import torch

from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel



@st.cache(persist=True, suppress_st_warning=True)
def load_corpus(data_files, ext_type='csv'):
    '''
    '''
    return load_dataset(ext_type, data_files=data_files)


def filter_corpus(corpus, element, filter_with):
    '''
    '''
    return corpus.filter(lambda filtered_corpus: filtered_corpus[element].startswith(filter_with))


@st.cache(persist=True, suppress_st_warning=True)
def get_script_episode(corpus, season, episode):
    '''
    '''
    data_season = filter_corpus(corpus, 'Season', season)
    return filter_corpus(data_season, 'Episode', episode)


@st.cache(persist=True, suppress_st_warning=True)
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


@st.cache(suppress_st_warning=True)
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


@st.cache(suppress_st_warning=True)
def get_similar_sentences(sentence, other_sentences, similarity_threshold=0.7):
    '''
    '''
    sims = cosine_similarity(sentence, other_sentences)
    return np.where(sims.reshape(-1) >= similarity_threshold)[0]


@st.cache(suppress_st_warning=True)
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


def display_scatterplot_3D(embeddings, sentences, user_input=None, label=None, color_map=None, annotation='On',
                           dim_red='PCA', perplexity=0, learning_rate=0, iteration=0, topn=0, sample=10):
    '''
    https://github.com/marcellusruben/Word_Embedding_Visualization
    '''

    if dim_red == 'PCA':
        three_dim = PCA(random_state=0).fit_transform(embeddings)[:, :3]
    else:
        three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                         n_iter=iteration).fit_transform(embeddings)[:, :3]

    color = 'blue'
    quiver = go.Cone(
        x=[0, 0, 0],
        y=[0, 0, 0],
        z=[0, 0, 0],
        u=[0.5, 0, 0],
        v=[0, 0.5, 0],
        w=[0, 0, 0.5],
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False
    )

    data = [quiver]

    count = 0

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=sentences[count:],
        name='input words',
        textposition="top center",
        textfont={
            'size': 15,
            'color': 'blue'
        },
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    data.append(trace_input)

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
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


def display_scatterplot_2D(embeddings, sentences, user_input=None, label=None, color_map=None, annotation='On',
                           dim_red='PCA', perplexity=0, learning_rate=0, iteration=0, topn=0, sample=10):
    '''
    https://github.com/marcellusruben/Word_Embedding_Visualization
    '''
    if dim_red == 'PCA':
        two_dim = PCA(random_state=0).fit_transform(embeddings)[:, :2]
    else:
        two_dim = TSNE(random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                       n_iter=iteration).fit_transform(embeddings)[:, :2]

    data = []
    count = 0

    trace_input = go.Scatter(
        x=two_dim[count:, 0],
        y=two_dim[count:, 1],
        text=sentences[count:],
        name='Sentence',
        textposition="top center",
        textfont_size=10,
        mode='markers+text',
        marker={
            'size': 15,
            'opacity': 0.8,
            'color': 'black'
        }
    )

    data.append(trace_input)

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
    # mytokens = [word.lemma_.lower().strip() if word.lemma_ != "PRON" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in constants.STOPWORDS and word not in constants.PUNCTUATIONS]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


def get_LDA_visualizer(data, topics, mds="tsna"):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    tokens = spacy_tokenizer(data)
    data_vectorized = vectorizer.fit_transform([tokens])
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=topics, max_iter=10, learning_method='online', verbose=True)
    data_lda = lda.fit_transform(data_vectorized)
    # Keywords for topics clustered by Latent Dirichlet Allocation
    dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds=mds)
    html = pyLDAvis.prepared_data_to_html(dash)
    return html
