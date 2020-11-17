import nltk
import numpy as np
import fasttext
from nltk.corpus import stopwords

# initialize the word embeddings

# bert_embedding = DocumentPoolEmbeddings([BertEmbeddings()])
# glove_embedding = DocumentPoolEmbeddings([WordEmbeddings('glove')])

# bert_embedding = BertEmbeddings()
# glove_embedding = WordEmbeddings('glove')

# flair_embedding_forward = FlairEmbeddings('news-forward')
# flair_embedding_backward = FlairEmbeddings('news-backward')
# stacked_embeddings = StackedEmbeddings([glove_embedding,
#                                       flair_embedding_backward,
#                                       flair_embedding_forward])

nltk.download('stopwords')

embedding_sizes = {'glove': 100, 'bert': 3072, 'flair_stacked': 5000, 'fasttext': 300}

embedding_methods = {
    # 'glove' : glove_embedding,
    # 'bert' : bert_embedding,
    'fasttext': None
}

fasttext_model = None

grammar = r"""
            NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            VP:
            {<V.*>}  # terminated with Verbs
            NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """

cp = nltk.RegexpParser(grammar)

english_stopwords = stopwords.words('english')


def load_fasttext(path: str):
    global fasttext_model
    fasttext_model = fasttext.load_model(path)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def embed_sentence(target_phrase, normalize=True, embedding_method_name='fasttext'):
    global fasttext_model
    if not fasttext_model:
        raise RuntimeError(
            "The fasttext model needs to be initialized first. For this call 'load_fasttext' of utils.py.")
    embedding_size = embedding_sizes[embedding_method_name]

    if embedding_method_name == 'fasttext':
        # to lower
        target_phrase = target_phrase.lower()
        # remove stop words
        target_phrase_tokens = [x for x in target_phrase.split(' ') if x not in english_stopwords]
        vectors = [fasttext_model[token] for token in target_phrase_tokens]

        if len(target_phrase_tokens) == 0:
            print('WARNING: trying to embed empty list of tokens.. returning default random vector')
            return np.random.uniform(0, 1, size=embedding_size).astype(np.float32)

        if normalize:
            vectors = [
                vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else np.zeros(embedding_size, dtype=np.float32)
                for vec in vectors]

        avg_embedding = np.mean(vectors, axis=0)

        return avg_embedding
