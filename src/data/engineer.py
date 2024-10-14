import pandas as pd
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import torch


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, bert_model_name: str = 'bert-base-uncased', test: bool = False):
        """
        Initializes the FeatureEngineer class with the dataset

        :param df: DataFrame
            DataFrame corresponding to the dataset
               bert_model_name: str
            The BERT model to use for extracting features (default: 'bert-base-uncased')
        """
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.test = test

    def extract_bert_embeddings(self, text: str) -> torch.Tensor:
        """
        Extract BERT embeddings for a given text, with attention to padding.

        :param text: str
            The input text
        :return: torch.Tensor
            BERT embeddings for the input text
        """
        # Tokenize the input text (can also handle batch input by passing a list of strings)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the embeddings from the last hidden layer
        last_hidden_state = outputs.last_hidden_state

        # Get the attention mask to exclude padding from the mean pooling
        attention_mask = inputs['attention_mask']

        # Perform mean pooling, excluding the padded tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)

        # Avoid division by zero for empty sequences
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Compute the mean pooled embedding
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.squeeze()

    def engineer_features(self):
        """
        Method for feature engineering of the dataset

        :return: DataFrame
            Feature engineered DataFrame
        """

        # Apply BERT embeddings to each tweet
        self.df['bert_embeddings'] = self.df['clean_text'].apply(self.extract_bert_embeddings)
        self.df['bert_embeddings'] = torch.stack(self.df['bert_embeddings'].tolist())

        # Add sentiment feature using TextBlob's sentiment analysis model
        self.df['sentiment'] = self.df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Drop irrelevant or redundant column features
        drop_columns = ['location', 'text', 'clean_text']
        if not self.test:
            drop_columns.append('id')

        self.df.drop(columns=drop_columns, axis=1, inplace=True)

        return self.df
