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


def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float, price_col: str = 'close', trading_days: int = 252) -> float:
    """
    Calculates the Sharpe Ratio (SR).
    Formula: SR = (Mean(R) - Rf) / StdDev(R) 
    """

    calc_daily_returns(df, price_col)
    excess_returns = (df['daily_return'] - risk_free_rate )
    avg_excess_returns = excess_returns.mean()
    std_excess_returns = excess_returns.std()
    df.drop(columns=['daily_return'], inplace=True)  # Clean up temporary column

    return (avg_excess_returns / std_excess_returns) * np.sqrt(trading_days)

# Tail-Risk & Loss Metrics

def calculate_max_drawdown(df: pd.DataFrame, price_col: str = 'close') -> float:
    """
    Calculates the Maximum Drawdown (MDD).
    Formula: MDD = (Peak - Trough) / Peak
    """
    df['cumulative_max'] = df[price_col].cummax()
    df['drawdown'] = (df[price_col] - df['cumulative_max']) / df['cumulative_max']
    max_drawdown = df['drawdown'].min()
    df.drop(columns=['cumulative_max', 'drawdown'], inplace=True)  # Clean up temporary columns
    return max_drawdown  

def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float, price_col: str = 'close') -> float:
    """
    Calculates the Sortino Ratio (SR).
    Formula: SR = (Mean(R) - Rf) / DownsideDeviation(R)
    """
    calc_daily_returns(df, price_col)
    avg_returns = df['daily_return'].mean()
    df['downside_returns'] = df['daily_return'].where(df['daily_return'] < risk_free_rate)
    std_downside = df['downside_returns'].std()

    df.drop(columns=['daily_return', 'downside_returns'], inplace=True)  # Clean up temporary columns
    
    return (avg_returns - risk_free_rate) / std_downside 



def calc_SMA(df: pd.DataFrame, window: int = 50, col: str = 'close') -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA) for a given window size.
    """
    df[f'SMA_{window}'] = df[col].rolling(window=window).mean()
    return df

 