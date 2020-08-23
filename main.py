import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from spacy.lang.pt import Portuguese
from spacy.lang.es import Spanish

# Constants - Hyperparameters
interactions_scores_dict = {'VIEW': 1, 'BOOKMARK': 2, 'FOLLOW': 3, 'LIKE': 4, 'COMMENT CREATED': 5}

# Global objects
interactions_df = pd.read_csv('interactions.csv')
articles_df = pd.read_csv('articles.csv')
person_le = preprocessing.LabelEncoder()
article_le = preprocessing.LabelEncoder()
interactions_matrix = None
hidden_dimensions = 20
language_objects = {"en": English(), "pt": Portuguese(), "es": Spanish()}
tokenizers = {}


# def create_interaction_matrix():
#     interactions_df = pd.read_csv('interactions.csv')
#     articles_df = pd.read_csv('articles.csv')
#     # User ids
#     person_le.fit(interactions_df.personId.unique())
#     # Articles ids
#     article_le.fit(articles_df.contentId.unique())
#     # Create interactions matrix
#     user_count = len(person_le.classes_)
#     article_count = len(article_le.classes_)
#     interactions_matrix = np.zeros(shape=(user_count, article_count), dtype=np.float32)
#     # Fill the interaction matrix: For every interaction entry in
#
#     print("X")

# We create tokens for articles based on tf-idf analysis of the 1-gram (single words)
# We are going to use Spacy's tokenizer for the corresponding language


def create_article_tokens():
    def identity_tokenizer(text):
        return text

    tf_idf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words="english",
                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,
                                        ngram_range=(1, 2), lowercase=False)
    corpus = []
    for index, row in articles_df.iterrows():
        language = row["lang"]
        # If we don't support the language, fall back to English.
        if language not in language_objects:
            nlp_object = language_objects["en"]
        else:
            nlp_object = language_objects[language]
        # Create the tokenizer for the language if it doesn't exists.
        if language not in tokenizers:
            tokenizers[language] = nlp_object.Defaults.create_tokenizer(nlp_object)
        tokenizer = tokenizers[language]
        # Tokenize both the article title and text
        title_text = row["title"]
        article_text = row["text"]
        title_text = title_text.lower().replace("\n", " ")
        article_text = article_text.lower().replace("\n", " ")
        title_tokens = tokenizer(title_text)
        article_tokens = tokenizer(article_text)
        title_tokens = [tk.text for tk in title_tokens if tk.text != "" and len(tk.text) > 1]
        article_tokens = [tk.text for tk in article_tokens if tk.text != "" and len(tk.text) > 1]
        all_tokens = []
        all_tokens.extend(title_tokens)
        all_tokens.extend(article_tokens)
        corpus.append(all_tokens)
        print("Article {0} has been processed.".format(index))
    # Transform words; apply tf-idf transformer
    tf_idf_vectorizer.fit(corpus)
    print("X")


if __name__ == "__main__":
    # create_interaction_matrix()
    create_article_tokens()
