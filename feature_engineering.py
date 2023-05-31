import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Useful text-preprocessing commands
from flair.data import Sentence
from flair.models import TextClassifier

CLASSIFIER = TextClassifier.load('sentiment-fast')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from utils import *
from sklearn_pandas import DataFrameMapper


def reviews_processed(df):
    """
         A function to create the new column:reviews_processed and apply text normalization and standardization
    input: Dataframe
    return: DataFrame with a new column contains processed text
    """
    df['reviews_processed'] = [word.lower() for word in df[
        'review_body']]  # all characteres as lower-case letters and assign them to a new column:reviews_processed
    lemmatizer = WordNetLemmatizer()
    for i in range(df.shape[0]):
        # split the words and keep alpha-numerical values and remove non-informative words(stop-words)
        m = [w for w in word_tokenize(df.loc[i, 'reviews_processed']) if
             str.isalpha(w) and w not in set(stopwords.words('english'))]
        # join the words into a string after lemmatizing them
        df.loc[i, 'reviews_processed'] = ' '.join([lemmatizer.lemmatize(w) for w in m])
    # drop the rows that their review_body is an empty string
    n_rows = list(df[df['reviews_processed'] != ''].index)
    return df.loc[n_rows]


def generate_sentiment(df, loop_number: int):
    """
         A function to generate sentiment over a given number of loops,
         that use a pre-trained model to predict if the review text is positive or negative
    df: DataFrame
    loop_number: int
    return: DataFrame with columns: ['reviews', 'sentiment', 'score'], loop-number the time of execution
    """
    flair_df = pd.DataFrame(columns=['reviews', 'sentiment', 'score'])
    reviews_list = list(df.reviews_processed[0:loop_number])
    tic()
    for i in range(loop_number):
        text = reviews_list[i]
        sentence = Sentence(text)
        CLASSIFIER.predict(sentence)
        flair_df.loc[i] = [text, str(sentence.labels[0]).split()[0], str(sentence.labels[0]).split()[1][1:-1]]
    delta = toc()
    return flair_df, [loop_number, delta]


def batch(iterable, mini_batch: int):
    """
        batch function that takes iterable(dataset) and a size(mini_batch)
    iterable: pd.array
    mini_batch: int
    """
    n = len(iterable)  # number of batches
    for ndx in range(0, n, mini_batch):
        # yields out the mini-batches of the iterable
        yield iterable[ndx:min(ndx + mini_batch, n)]


def get_sentiment(text_array):
    """
        get_sentiment function for batches
    text_array: np.array
    return: text np.array of predicted values of this text
    """
    n = len(text_array)
    sentence_list = [Sentence(text_array[i]) for i in range(n)]
    CLASSIFIER.predict(sentence_list, mini_batch_size=128, verbose=True)
    sentence_labels = [sentence.labels for sentence in sentence_list]
    return np.array(sentence_labels)


def apply_batch(df, batch_size):
    """
        a function that applies the minibatch method on a given data set and reserves the sentiment
        value and score as columns in this data frame

    df: DataFrame
    batch_size: int
    return: DataFrame with the columns ['sent_score', 'sent_value']
    """
    value, score = [], []
    # iter over batches and apply get_sentiment on each
    for x in batch(df.reviews_processed, batch_size):
        label = get_sentiment(np.array(x)).astype(str)
        # print(len(label))
        # append sentiment value and score to the input data frame
        for j in range(len(label)):
            value.append(np.char.split(label)[j][0][0])
            score.append(float(np.char.split(label)[j][0][1][1:-1]))

    df['sent_value'], df['sent_score'] = [value, score]
    return df


def pre_processing(df):
    """
        A function that simplify the target variable and get a binary classification problem,
        it maps star ratings of 1,2 to 0, and 4,5 to 1
        (corresponding to negative and positive sentiments respectively).
        Filter neutral star ratings of 3
    df: DataFrame
    return: processed DataFrame with column ['binstar']
    """
    # Set string variables values into numeric
    cols_to_numeric = ['customer_id', 'product_parent', 'star_rating', 'helpful_votes', 'total_votes']
    df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric)
    df['review_date'] = pd.to_datetime(df['review_date'])

    # map star ratings into new variable binstar
    conditions = [
        (df.star_rating.isin([0, 1, 2])),
        (df.star_rating.isin([4, 5])),
        (df.star_rating == 3)]
    choices = [0, 1, np.nan]
    df['binstar'] = np.select(conditions, choices, default=np.nan)

    # Set binary variables values to zero/one
    df['sent_value'] = np.select([(df.sent_value == 'POSITIVE'), (df.sent_value == 'NEGATIVE')],
                                 [1, -1], default=np.nan).astype(int)
    df['vine'] = np.select([(df.vine == 'Y'), (df.vine == 'N')],
                           [1, 0], default=np.nan)
    df['verified_purchase'] = np.select([(df.verified_purchase == 'Y'), (df.verified_purchase == 'N')],
                                        [1, 0], default=np.nan)

    # choose columns for modeling:
    cols_for_modeling = ['reviews_processed', 'helpful_votes', 'total_votes', 'sent_score', 'sent_value', 'binstar']
    df = df[cols_for_modeling]

    # Get rid of np.nan, np.inf, -np.inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)

    df['binstar'] = df['binstar'].astype(int)
    return df


def transform(df):
    """
        A function that add the features of the dataframe that you want to transform and/or combine
    df: DataFrame
    return: Transformed DataFrame
    """
    mapper = DataFrameMapper([
        ('reviews_processed', TfidfVectorizer(max_features=100)),
        ('helpful_votes', None),
        ('total_votes', None),
        ('sent_score', None),
        ('sent_value', None)
    ], df_out=False)

    """
    Use the fit_transform method to transform the old dataframe into a new one
    that can be fed to the machine learning algorithm.
    """
    mapper_fit = mapper.fit(df.iloc[:, :-1])
    final_df = mapper.transform(df.iloc[:, :-1])  # a numpy array
    return final_df
