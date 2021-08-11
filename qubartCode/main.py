from Utils import Utils
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    # Loading data
    utils = Utils()

    corpus = utils.load_corpus("corpus/Game_Of_Thrones_Script.csv")['train']
    s1_ep2 = utils.get_full_text(utils.get_script_episode(corpus, 'Season 1', 'Episode 2')['Sentence'])
    s1_ep2_names = ''
    for row in utils.get_script_episode(corpus, 'Season 1', 'Episode 2'):
        line = f"{row['Name'].capitalize()}: {row['Sentence']}"
        s1_ep2_names = s1_ep2_names + " " + line
    corpus_reviews = utils.load_corpus("corpus/GameOfThrones_Reviews.csv")['train']


    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)
    print(sentence_embeddings.shape)

