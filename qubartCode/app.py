import streamlit as st
from Utils import *
import re

class select_box:
    def __init__(self, data):
        self.data = data
        self.box = None

    def place(self, key):
        self.box = st.selectbox(str(key), self.data)
        select_box.value = self.box


def title(text, size, color="white"):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',
                unsafe_allow_html=True)


def header(text):
    st.markdown(f"<p style='color:white;'>{text}</p>", unsafe_allow_html=True)


title("Game Of Thrones Summary Generator", 60)

corpus_selected = st.sidebar.selectbox("Select desired corpus", ["Reviews", "Episode scripts"])

corpus = None

text = ""

if corpus_selected == "Episode scripts":
    corpus = load_corpus("corpus/Game_Of_Thrones_Script.csv")['train']
    season = st.sidebar.selectbox("Select season: ", [1, 2, 3, 4, 5, 6, 7, 8], key="season")
    episode = st.sidebar.selectbox("Select episode:", [1, 2, 3, 4, 5, 6, 7, 8], key="episode")

    episode_got = get_script_episode(corpus, f'Season {season}', f'Episode {episode}')

    title(episode_got["Episode Title"][0], 30, "pink")

    episode_got_names = ""
    for row in episode_got:
        line = f"{row['Name'].capitalize()}: {row['Sentence']}"
        episode_got_names = episode_got_names + "{}\r\n".format(line)

    sentences = episode_got_names.split("\r\n")
    expander = st.expander("Show Episode")
    expander.write(episode_got['Sentence'])

else:
    corpus = load_corpus("corpus/GameOfThrones_Reviews.csv")['train']
    review = st.sidebar.selectbox("Select episode review: ", [f"Episode {i + 1}" for i in range(0,72)], key="season")
    title(corpus['Episode title'][int(re.search(r'\d+', review).group()) - 1], 30, "pink")
    episode_got = corpus['Episode recap'][int(re.search(r'\d+', review).group()) - 1]
    sentences = episode_got.split(".")
    expander = st.expander("Show Episode Review")
    expander.write(episode_got)

model_selected = st.sidebar.selectbox("Select desired model", ["sshleifer/distilbart-cnn-12-6",
                                                               "sshleifer/distilbart-xsum-12-3",
                                                               "google/pegasus-xsum"])

title("Filter text", 40, "pink")

embedded_sentences = get_sentence_embeddings(sentences)

filter_by = st.text_input("Filter with", "Bran")
"Filtered with", filter_by
similarity_threshold = st.slider('Similarity threshold', 0.0, 1.0, 0.3)

similar_sentences = get_similar_sentences(get_sentence_embeddings(filter_by), embedded_sentences, similarity_threshold=similarity_threshold)
filtered_text = ""
for idx in similar_sentences:
    filtered_text = filtered_text + " " + sentences[idx]
title("Filtered text", 30, "pink")
filtered_text


title("Summary", 40, "pink")

summary = get_summary(filtered_text, model_selected)
summary[0]
