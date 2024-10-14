import pandas as pd
from sklearn.preprocessing import LabelEncoder

import string


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Preprocessor class with the dataset

        :param df: DataFrame
            DataFrame corresponding to the dataset
        """
        self.df = df

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Method for cleaning text

        :param text: str
            Raw text
        :return: str
            Clean text
        """

        # Get rid of punctuation and convert the text to lower case
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()

        return text

    def preprocess(self):
        """
        Preprocesses the dataset before training

        :return: DataFrame
            Preprocessed DataFrame
        """

        # Fill missing values for keyword with 'unknown'
        self.df['keyword'] = self.df['keyword'].fillna('unknown')

        # Use label encoding for keyword column in the dataset
        encoder = LabelEncoder()
        self.df['keyword'] = encoder.fit_transform(self.df['keyword'])

        # Clean the tweet text
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)

        return self.df
