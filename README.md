# Disaster Tweets Classifier

This project is a submission for Kaggle's [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition. The goal is to predict whether a tweet is about a real disaster or not using machine learning techniques.

## Project Overview

The dataset consists of tweets labeled as either disaster-related or not. The main focus of this project was to preprocess the data, extract features using BERT embeddings, and build a binary classification model using PyTorch.

## Key Features

- **BERT Embeddings**: Extracting deep contextualized embeddings from the tweets using a pre-trained BERT model.
- **Feature Engineering**: Additional features like sentiment analysis were added to enhance the prediction power of the model.
- **PyTorch Model**: A neural network was used for binary classification.

## Setup

1. Clone the repository:
   
```bash
git clone https://github.com/yourusername/disaster-tweets-classifier.git
cd disaster-tweets-classifier
```

2. Install dependencies:
   
```bash
pip install -r requirements.txt
```

3. Run the main script:

```bash
python main.py
```

## Data

The dataset is sourced from the competition's [Kaggle page](https://www.kaggle.com/competitions/nlp-getting-started). Please download the dataset from there and place it in the appropriate folder.
