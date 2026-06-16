import numpy as np
import pandas as pd


# Return Metrics 
def calc_daily_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculates daily percentage returns using vectorized array operations.
    Formula: R_t = (P_t / P_{t-1}) - 1
    """
    df['daily_return'] = df[price_col].pct_change()
    return df

def calc_cumulative_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculates cumulative returns using vectorized array operations.
    Formula: C_t = (P_t / P_0) - 1
    """
    df['cumulative_return'] = (df[price_col] / df[price_col].iloc[0]) - 1
    return df


def calc_annualized_return(df: pd.DataFrame, price_col: str = 'close', trading_days: int = 252) -> float:
    """
    Calculates annualized returns (CAGR).
    Formula: AR = (EV/SV) ^ 1/n - 1 where n = rows/trading_days
    """
    ev = df[price_col].iloc[-1]
    sv = df[price_col].iloc[0]

    return (ev / sv) ** (trading_days / len(df)) - 1


# Risk Metrics
def calculate_annualized_volatility(df: pd.DataFrame, price_col: str = 'close', trading_days: int = 252, crypto: bool = False) -> float:
    """
    Calculates annualized volatility (Vol).
    Formula: Vol = StdDev(R) * sqrt(trading_days)
    """
    if crypto: 
        trading_days = 365  # Adjust for crypto markets
    
    period_returns = np.log( df[price_col] / df[price_col].shift(1) )
    return period_returns.std() * np.sqrt(trading_days)


# Tail-Risk & Loss Metrics

def calc_SMA(df: pd.DataFrame, window: int = 50, col: str = 'close') -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA) for a given window size.
    """
    df[f'SMA_{window}'] = df[col].rolling(window=window).mean()
    return df

 