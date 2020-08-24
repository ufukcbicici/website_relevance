import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from spacy.lang.pt import Portuguese
from spacy.lang.es import Spanish
import gensim
import re
import os
import pickle
from collections import Counter

# Constants - Hyperparameters
interactions_scores_dict = {'VIEW': 1, 'BOOKMARK': 2, 'FOLLOW': 3, 'LIKE': 4, 'COMMENT CREATED': 5}

# Global objects
interactions_df = pd.read_csv('interactions.csv')
articles_df = pd.read_csv('articles.csv')
person_le = preprocessing.LabelEncoder()
tokens_le = preprocessing.LabelEncoder()
hidden_dimensions = 250
language_objects = {"en": English(), "pt": Portuguese(), "es": Spanish()}
tokenizers = {}
summaries = {}
filter_regex = "[^A-Za-z0-9]+"
batch_size = 10000
max_iterations = 100000
l2_lambda = 0.0


# We summarize each article with Spacy's TextRank implementation. This eliminates most of the noisy information
# in the texts. Then we apply tf-idf analysis to the article summaries. For every unique token in the obtained corpus
# of summaries, we calculate the expected tf-idf score over all articles. Then we sort the tokens in descending order
# of their expected tf-idf scores. The first 5000 tokens will constitute the representing tokens of our article corpus.


def create_article_tokens():
    def identity_tokenizer(text):
        return text

    if os.path.isfile("selected_tokens.sav"):
        f = open(os.path.join("selected_tokens.sav"), "rb")
        final_tokens = pickle.load(f)
        f.close()
        return final_tokens

    tf_idf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words="english",
                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                        ngram_range=(1, 1), lowercase=False)
    corpus = []
    for index, row in articles_df.iterrows():
        language = row["lang"]
        article_id = row["contentId"]
        # if article_id == 5714314286511882372:
        #     print("X")
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
        summary = summary.lower().replace("\n", " ").replace("'", "")
        summary_tokens = tokenizer(summary)
        filtered_summary_tokens = []
        # Process tokens with regex; remove all special characters
        for tk in summary_tokens:
            original_token_text = tk.text
            filtered_text = re.sub(filter_regex, '', original_token_text)
            # print("filtered_text={0}".format(filtered_text))
            # print(len(filtered_text))
            if len(filtered_text) > 1:
                filtered_summary_tokens.append(filtered_text)
            # summary_tokens = [tk.text for tk in summary_tokens if tk.text != "" and len(tk.text) > 1]
        summaries[article_id] = filtered_summary_tokens
        corpus.append(filtered_summary_tokens)
        print("Article {0} has been processed.".format(index))
    # Transform words; apply tf-idf transformer
    feature_matrix = tf_idf_vectorizer.fit_transform(corpus)
    # Calculate mean tf-idf scores for all n-grams over all articles.
    tf_idf_scores = np.mean(feature_matrix, axis=0)
    feature_names = tf_idf_vectorizer.get_feature_names()
    feature_scores = [(feature_names[idx], tf_idf_scores[0, idx]) for idx in range(tf_idf_scores.shape[1])]
    sorted_features = sorted(feature_scores, key=lambda tpl: tpl[1], reverse=True)
    final_tokens = [tpl[0] for tpl in sorted_features[0:5000]]
    # Save the tokens
    f = open(os.path.join("selected_tokens.sav"), "wb")
    pickle.dump(final_tokens, f)
    f.close()
    return final_tokens


def create_interaction_matrix():
    if os.path.isfile("interactions_matrix.sav"):
        f = open(os.path.join("interactions_matrix.sav"), "rb")
        matrix = pickle.load(f)
        f.close()
        return matrix

    token_set = set(selected_tokens)
    # User ids
    person_le.fit(interactions_df.personId.unique())
    # Token ids
    tokens_le.fit(selected_tokens)
    # Create interactions matrix
    user_count = len(person_le.classes_)
    article_count = len(tokens_le.classes_)
    interactions_matrix = np.zeros(shape=(user_count, article_count), dtype=np.float32)
    # Fill the interaction matrix: For every interaction entry in
    for index, row in interactions_df.iterrows():
        if (index + 1) % 1000 == 0:
            print("{0} rows have been processed.".format(index))
        person_id = row["personId"]
        person_index = person_le.transform([person_id])[0]
        article_id = row["contentId"]
        event_type = row["eventType"]
        if article_id not in summaries:
            continue
        summary = summaries[article_id]
        valid_tokens = [tk for tk in summary if tk in token_set]
        token_indices = tokens_le.transform(valid_tokens)
        score = interactions_scores_dict[event_type]
        for token_index in token_indices:
            interactions_matrix[person_index, token_index] += score
    f = open(os.path.join("interactions_matrix.sav"), "wb")
    pickle.dump(interactions_matrix, f)
    f.close()
    return interactions_matrix


def apply_matrix_factorization(V_):
    # Interaction matrix
    with tf.variable_scope("MF_Regressor"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        v_ = tf.constant(V_)
        # User matrix
        W_ = tf.get_variable(trainable=True,
                             initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1),
                             shape=(V_.shape[0], hidden_dimensions),
                             name="user_matrix")
        # Token matrix
        H_ = tf.get_variable(trainable=True,
                             initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1),
                             shape=(hidden_dimensions, V_.shape[1]),
                             name="token_matrix")
        # User bias
        b_w = tf.get_variable(trainable=True,
                              initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1),
                              shape=(V_.shape[0],),
                              name="user_bias")
        # Token bias
        b_h = tf.get_variable(trainable=True,
                              initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1),
                              shape=(V_.shape[1],),
                              name="token_bias")
        # Mean bias
        mu = tf.get_variable(trainable=True,
                             initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.1),
                             shape=(1,),
                             name="mean_bias")
        # Selected user and token indices
        user_indices = tf.placeholder(dtype=tf.int32, shape=[None])
        token_indices = tf.placeholder(dtype=tf.int32, shape=[None])
        regularizer_strength = tf.placeholder(dtype=tf.float32)
        indices = tf.stack([user_indices, token_indices], axis=1)
        # Get non zero ratings from the interaction matrix
        ratings = tf.gather_nd(params=v_, indices=indices)
        # Get feature vectors for users
        user_features = tf.gather_nd(params=W_, indices=tf.expand_dims(user_indices, axis=1))
        token_features = tf.gather_nd(params=tf.transpose(H_), indices=tf.expand_dims(token_indices, axis=1))
        user_biases = tf.gather_nd(params=b_w, indices=tf.expand_dims(user_indices, axis=1))
        token_biases = tf.gather_nd(params=b_h, indices=tf.expand_dims(token_indices, axis=1))
        # Estimated ratings
        unbiased_rating_estimates = tf.reduce_sum(tf.multiply(user_features, token_features), axis=1)
        # Final estimations
        estimated_ratings = unbiased_rating_estimates + user_biases + token_biases + mu
        norms = tf.norm(user_features, ord="euclidean") + tf.norm(token_features, ord="euclidean") + \
                tf.norm(user_biases, ord="euclidean") + tf.norm(token_biases, ord="euclidean")
        loss = tf.reduce_mean(tf.square(ratings - estimated_ratings)) + regularizer_strength * norms
        # Estimated matrix
        estimated_matrix = tf.linalg.matmul(W_, H_)
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Indices for the non-zero entries of the interactions matrix
    non_zero_rating_indices = np.stack(np.nonzero(V_), axis=1)
    losses = []
    for iteration in range(max_iterations):
        # Draw random minibatches from existing ratings
        minibatch_indices = np.random.choice(non_zero_rating_indices.shape[0], batch_size, replace=True)
        selected_rating_indices = non_zero_rating_indices[minibatch_indices]
        results = sess.run([loss, optimizer],
                           feed_dict={user_indices: selected_rating_indices[:, 0],
                                      token_indices: selected_rating_indices[:, 1],
                                      regularizer_strength: l2_lambda})
        losses.append(results[0])
        if iteration % 10 == 0:
            mean_loss = np.mean(np.array(losses))
            print("Iteration:{0} mean_loss={1}".format(iteration, mean_loss))
            losses = []
    # Reconstruct the dense matrix
    res = sess.run([estimated_matrix])
    estimated_interactions_matrix = res[0]
    assert estimated_interactions_matrix.shape == V_.shape
    f = open(os.path.join("estimated_interactions_matrix.sav"), "wb")
    pickle.dump(estimated_interactions_matrix, f)
    f.close()


if __name__ == "__main__":
    # create_interaction_matrix()
    selected_tokens = create_article_tokens()
    interactions_matrix = create_interaction_matrix()
    apply_matrix_factorization(interactions_matrix)
