import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from spacy.lang.pt import Portuguese
from spacy.lang.es import Spanish
import gensim
from collections import Counter

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
selected_tokens = []

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
                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                        ngram_range=(1, 1), lowercase=False)
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
        # Calculate the text summary with Gensim's TextRank implementation.
        whole_text = row["title"] + "\n" + row["text"]
        summary = gensim.summarization.summarize(text=whole_text, ratio=0.1, split=False)
        summary = summary.lower().replace("\n", " ")
        summary_tokens = tokenizer(summary)
        summary_tokens = [tk.text for tk in summary_tokens if tk.text != "" and len(tk.text) > 1]
        corpus.append(summary_tokens)
        print("Article {0} has been processed.".format(index))
    # Transform words; apply tf-idf transformer
    feature_matrix = tf_idf_vectorizer.fit_transform(corpus)
    # Calculate mean tf-idf scores for all n-grams over all articles.
    tf_idf_scores = np.mean(feature_matrix, axis=0)
    feature_names = tf_idf_vectorizer.get_feature_names()
    feature_scores = [(feature_names[idx], tf_idf_scores[0, idx]) for idx in range(tf_idf_scores.shape[1])]
    sorted_features = sorted(feature_scores, key=lambda tpl: tpl[1], reverse=True)
    final_tokens = [tpl[0] for tpl in sorted_features[0:5000]]
    return final_tokens


if __name__ == "__main__":
    # create_interaction_matrix()
    create_article_tokens()
