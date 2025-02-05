import pandas as pd

def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    return df['Close'].values
