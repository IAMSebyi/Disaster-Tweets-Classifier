import pandas as pd

from src.data import DataLoader, FeatureEngineer, Preprocessor
from src.models import train_model, evaluate_model

import torch

from sklearn.model_selection import train_test_split

from pathlib import Path
import joblib


def main():
    # Check if processed datasets exist
    if Path("data/processed/train").exists() and Path("data/processed/test").exists():
        print("Loading processed dataset...")
        train_df = joblib.load("data/processed/train")
        test_df = joblib.load("data/processed/test")
    else:
        # Define paths
        train_path = Path("data/raw/train.csv")
        test_path = Path("data/raw/test.csv")

        # Load training and test datasets
        print("Loading raw dataset...")
        train_loader = DataLoader(train_path)
        train_df = train_loader.load_data()
        test_loader = DataLoader(test_path)
        test_df = test_loader.load_data()

        # Preprocess datasets
        print("Applying preprocessing...")
        train_preproc = Preprocessor(train_df)
        train_df = train_preproc.preprocess()
        test_preproc = Preprocessor(test_df)
        test_df = test_preproc.preprocess()

        # Feature engineer datasets
        print("Applying feature engineering...")
        train_engineer = FeatureEngineer(train_df)
        train_df = train_engineer.engineer_features()
        test_engineer = FeatureEngineer(test_df, test=True)
        test_df = test_engineer.engineer_features()

        # Save processed datasets
        print("Saving dataset...")
        joblib.dump(train_df, "data/processed/train")
        joblib.dump(test_df, "data/processed/test")

    # Extract features and labels from the training set and split
    print("Splitting training set...")
    X_train, X_eval, y_train, y_eval = train_test_split(train_df.drop(["target"], axis=1), train_df["target"], test_size=0.2, random_state=42)
    X_train, X_eval, y_train, y_eval = X_train.values, X_eval.values, y_train.values, y_eval.values

    # Train model on training set
    print("Training model...")
    input_size = X_train.shape[1]
    hidden_size = 128
    model = train_model(X_train, y_train, input_size, hidden_size)

    # Save model
    print("Saving model...")
    joblib.dump(model, "models/disaster_tweets_classifier")

    # Evaluate the model
    evaluate_model(model, X_eval, y_eval)

    # Get outputs for the test dataset
    inputs = torch.tensor(test_df.drop(["id"], axis=1).values, dtype=torch.float32)  # Ensure dtype is float32
    outputs = model(inputs)

    # Create a submission dataframe and save to CSV
    print("Saving predictions under submission.csv...")
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]
    submission["target"] = (outputs > 0.5).int()  # Binary classification (rounding at 0.5 threshold)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
