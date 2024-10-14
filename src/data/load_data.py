import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, filepath: Path):
        """
        Initializes the DataLoader class with the filepath to the dataset

        :param filepath: Path
            Path to the dataset
        """
        self.filepath = filepath

    def load_data(self):
        """
        Loads and returns dataset

        :return: DataFrame
            DataFrame corresponding to the dataset
        """
        try:
            return pd.read_csv(self.filepath)
        except FileNotFoundError as e:
            print(f"File not found: {self.filepath}")
            raise e
        except pd.errors.EmptyDataError:
            print("No data in the file.")
        except pd.errors.ParserError:
            print("Error parsing the file.")
