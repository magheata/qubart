from Utils import *
import streamlit as st

if __name__ == '__main__':
    # Loading data
    corpus = load_corpus("corpus/Game_Of_Thrones_Script.csv")['train']
    s1_ep2 = get_full_text(get_script_episode(corpus, 'Season 1', 'Episode 2')['Sentence'])
    s1_ep2_names = ''
    for row in get_script_episode(corpus, 'Season 1', 'Episode 2'):
        line = f"{row['Name'].capitalize()}: {row['Sentence']}"
        s1_ep2_names = s1_ep2_names + " " + line
    corpus_reviews = load_corpus("corpus/GameOfThrones_Reviews.csv")['train']


    sentences = corpus_reviews['Episode recap'][30].split('.')
    embedded_script_2 = get_sentence_embeddings(sentences)
    similar_sentences = get_similar_sentences(get_sentence_embeddings("Bran"), embedded_script_2, similarity_threshold=0.3)
    filtered_text = ""
    for idx in similar_sentences:
        filtered_text = filtered_text + " " + sentences[idx]
    print(filtered_text)

