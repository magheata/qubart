from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import streamlit as st


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