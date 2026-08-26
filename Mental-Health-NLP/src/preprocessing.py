import pandas as pd
import re



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_dataset(df):
    # Remove rows where text is missing
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def prepare_text(df):
    df["clean_text"] = df["text"].apply(clean_text)
    return df

