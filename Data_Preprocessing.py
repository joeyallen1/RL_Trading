import yfinance
import pandas as pd
import numpy as np
import sys
import os



def load_data(stock_ticker: str, start_date: str) -> pd.DataFrame:
    """Returns a dataframe containing the close and volume data 
    for a given stock ticker and start date, also drops rows with 0 or NaN values."""

    data = yfinance.download(stock_ticker, interval="1d", auto_adjust=True, start=start_date, multi_level_index=False)
    data = data.loc[:,['Close', 'Volume']]
    data.dropna(inplace=True)
    data = data[(data != 0).all(axis=1)]
    return data


def split_data(df: pd.DataFrame) -> tuple:
    """Splits the given dataframe into training, validation,
    and test sets."""

    length = len(df)
    training_df = df.iloc[:int(0.7 * length), :].copy(deep=True)
    validation_df = df.iloc[int(0.7*length):int(0.85*length), :].copy(deep=True)
    testing_df = df.iloc[int(0.85*length):, :].copy(deep=True)
    return training_df, validation_df, testing_df
    

def add_MACD(dataframes: tuple):
    """Adds column containing the MACD indicator and a column containing
    the MACD percentage indicator each given dataframe.
    
    MACD (moving average convergence/divergence) is a momentum
    indicator that uses two exponential moving averages of a stock's 
    price. The regular MACD is calculated by subtracting the long term
    EMA from the short term EMA, and then the regular MACD is divided
    by the long term EMA to get the MACD percentage (in terms of the 
    long term EMA). Here, the value is then scaled by a factor of 10
    to better align with the scaling of other features. 
    The MACD percentage is then clipped to the range [-1, 1].

    In practice, the regular MACD can be used
    as a buy signal when its value is greater than 0 and a sell
    signal when its value is less than 0."""

    for df in dataframes:
        df['EMA 12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA 26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA 12'] - df['EMA 26']
        df['MACD Percentage'] = ((df['EMA 12'] - df['EMA 26']) / df['EMA 26']) * 10
        df['MACD Percentage'] = np.clip(df['MACD Percentage'], -1.0, 1.0) 
        df.drop(labels=['EMA 12', 'EMA 26'], axis=1, inplace=True)


def add_Volume_Oscillator(dataframes: tuple):
    """Adds column containing the volume oscillator indicator
    to each given dataframe.
    
    The volume oscillator captures trends in the trading volume of a stock.
    It is calculated by subtracting a long term moving average of the trading
    volume from a short term moving average. This value is then divided
    by the long term average to get a percentage value, which is 
    then clipped to the range [-1, 1]."""

    for df in dataframes:
        df['EMA 5'] = df['Volume'].ewm(span=5, adjust=False).mean()
        df['EMA 20'] = df['Volume'].ewm(span=20, adjust=False).mean()
        df['Volume Oscillator'] = np.clip((df['EMA 5'] - df['EMA 20']) / df['EMA 20'], -1.0, 1.0) 
        df.drop(labels=['EMA 5', 'EMA 20'], axis=1, inplace=True)


def add_Coefficient_of_Variation(dataframes: tuple):
    """Adds column containing the coefficient of variation indicator to each given dataframe.
    
    The coefficient of variation of price can be used to measure the volatility of a stock. 
    The indicator used here is calculated using the standard deviation of prices divided
    by a simple moving average of prices over a rolling window. Here, the value is multiplied
    by a factor of 5 for scaling and then clipped to the range [0, 1]."""

    for df in dataframes:
        df['Std'] = df['Close'].rolling(window=10).std()
        df['SMA'] = df['Close'].rolling(window=10).mean()
        df.dropna(inplace=True)
        df['CV'] = df['Std'] / df['SMA'] * 5
        df['CV'] = np.clip(df['CV'], 0, 1)
        df.drop(labels=['Std', 'SMA'], axis=1, inplace=True)


def add_RSI(dataframes: tuple):
    """Adds column containing the RSI to the each given dataframe.
    
    RSI (Relative Strength Index) is a simple momentum indicator used
    to identify overbought or oversold conditions. It is calculated using the 
    formula 100 - (100 / (1 + R)) where R is the mean gain divided by the mean loss over a rolling window.
    In practice, a value above 70 is typically used to indicate that an asset is overbought 
    and a value less tahn 30 is typically used to indicate that an asset is oversold.
    Here, the value is then divided by 50 and then subtracted by 1 to get an indicator in the range [-1, 1]."""

    for df in dataframes:
        df['Diff'] = df['Close'].diff()
        df.dropna(inplace=True)
        df['Mean gain'] = df['Diff'].rolling(window=10).apply(lambda x: x[x > 0].mean())
        df['Mean loss'] = df['Diff'].rolling(window=10).apply(lambda x: abs(x[x <= 0].mean()))
        df.dropna(inplace=True)
        df['RSI'] = 100 - (100 / (1 + (df['Mean gain'] / df['Mean loss'])))
        df['RSI'] = df['RSI'] / 50 - 1
        df.drop(labels=['Diff', 'Mean gain', 'Mean loss'], axis=1, inplace=True)


def add_pct_change(dataframes: tuple):
    """Adds column containing the percentage change from the previous 
    close price to each given dataframe. This value is clipped to 
    the range [-1, 1]."""

    for df in dataframes:
        df['Pct Change'] = np.clip(df['Close'].pct_change(periods=5), -1, 1)
        df.dropna(inplace=True)


def drop_volume(dataframes: tuple):
    """Drops the volume column from all given dataframes."""

    for df in dataframes:
        df.drop(labels=['Volume'], axis=1, inplace=True)



# example:
# python Data_Preprocessing.py "KO" "2005-01-01"

if __name__ == "__main__":

    #TODO: revisit features/feature scaling

    stock_ticker = sys.argv[1]
    start_date = sys.argv[2]
    df = load_data(stock_ticker, start_date)
    training_df, validation_df, testing_df = split_data(df)
    dataframes = (training_df, validation_df, testing_df)
    add_MACD(dataframes)
    add_Volume_Oscillator(dataframes)
    add_Coefficient_of_Variation(dataframes)
    add_RSI(dataframes)
    add_pct_change(dataframes)
    drop_volume(dataframes)
    print(training_df.describe(), "\n")
    print(validation_df.describe(), "\n")
    print(testing_df.describe())
    try:
        os.mkdir(f"./{stock_ticker}")
    except FileExistsError:
        print("Cannot create directory since it already exists, please delete it manually first if you wish to overwrite it.")
    training_df.to_csv(f'./{stock_ticker}/Training.csv')
    validation_df.to_csv(f'./{stock_ticker}/Validation.csv')
    testing_df.to_csv(f'./{stock_ticker}/Testing.csv')