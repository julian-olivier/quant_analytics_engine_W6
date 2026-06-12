import pandas as pd
import numpy as np
import os

class LocalCSVLoader:
    def __init__(self, data_directory: str):
        """
        Initializes the loader with the directory path where your historical 
        financial CSV files are stored.
        """
        self.data_dir = data_directory

    def load_clean_data(self, ticker: str) -> pd.DataFrame:
        """
        Loads a market CSV file, enforces chronological sorting, standardizes 
        data schemas, and handles data gaps via forward-filling.
        
        Parameters:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            
        Returns:
            pd.DataFrame: A cleaned time-series dataframe indexed by Date.
        """
        # 1. Construct the target absolute or relative file path
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Execution Error: Raw data file not found at {file_path}")
        
        # 2. Ingest the data file using Pandas
        df = pd.read_csv(file_path)
        
        # 3. Standardize column headers to lowercase to prevent casing errors 
        # (e.g., handling 'Adj Close' vs 'adj close' vs 'Close' bugs)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure critical core columns exist in the raw dataset
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Data Schema Malformed: Missing required column '{col}' in {ticker}.csv")

        # 4. Enforce proper Datetime Formatting
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 5. CRITICAL STEP: Strict Chronological Sorting
        # Quant math (like rolling metrics) breaks if the data flows backwards or is randomized.
        df = df.sort_index(ascending=True)
        
        # 6. Clean Missing Data Points / Gaps
        # We use Forward-Fill ('ffill'). If a stock breaks or halts for a minute/day, 
        # it carries the last known trade price forward. 
        # Never Backward-Fill ('bfill') here, as that leaks future information into the past!
        df = df.ffill()
        
        # Drop any remaining NaN rows that couldn't be forward-filled (e.g., at the very start of the file)
        df = df.dropna()
        
        # 7. Explicitly Cast Data Types to optimize NumPy performance memory layout
        float_cols = ['open', 'high', 'low', 'close']
        if 'adj close' in df.columns:
            float_cols.append('adj close')
            
        df[float_cols] = df[float_cols].astype(np.float64)
        df['volume'] = df['volume'].astype(np.int64)
        
        return df