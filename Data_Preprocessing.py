import yfinance
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt



def load_data(stock_ticker: str, start_date: str) -> pd.DataFrame:
    """Returns a dataframe containing the close and volume data 
    for a given stock ticker and start date, also drops rows with 0 or NaN values."""

    data = yfinance.download(stock_ticker, interval="1d", auto_adjust=True, start=start_date, multi_level_index=False)
    data = data.loc[:,['Close', 'Volume']]
    data.dropna(inplace=True)
    data = data[(data != 0).all(axis=1)]
    return data


def split_data(df: pd.DataFrame) -> list:
    """Splits the given dataframe into training, validation,
    and test sets."""

    length = len(df)
    training_df = df.iloc[:int(0.7 * length), :].copy(deep=True)
    validation_df = df.iloc[int(0.7*length):int(0.85*length), :].copy(deep=True)
    testing_df = df.iloc[int(0.85*length):, :].copy(deep=True)
    return [training_df, validation_df, testing_df]
    

def add_MACD(dataframes: list):
    """Adds column containing the MACD indicator and a column containing
    the MACD percentage indicator to each given dataframe.
    
    MACD (moving average convergence/divergence) is a momentum
    indicator that uses two exponential moving averages of a stock's 
    price. The regular MACD is calculated by subtracting the long term
    EMA from the short term EMA, and then the regular MACD is divided
    by the long term EMA to get the MACD percentage (in terms of the 
    long term EMA). 

    In practice, the regular MACD can be used
    as a buy signal when its value goes from negative to positive
    and a sell signal when its value goes from positive to negative."""

    for df in dataframes:
        df['EMA 12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA 26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA 12'] - df['EMA 26']
        df['MACD Percentage'] = ((df['EMA 12'] - df['EMA 26']) / df['EMA 26'])
        df.drop(labels=['EMA 12', 'EMA 26'], axis=1, inplace=True)


def add_Volume_Oscillator(dataframes: list):
    """Adds column containing the volume oscillator indicator
    to each given dataframe.
    
    The volume oscillator captures trends in the trading volume of a stock.
    It is calculated by subtracting a long term moving average of the trading
    volume from a short term moving average. This value is then divided
    by the long term average to get a percentage value."""

    for df in dataframes:
        df['EMA 12'] = df['Volume'].ewm(span=12, adjust=False).mean()
        df['EMA 26'] = df['Volume'].ewm(span=26, adjust=False).mean()
        df['Volume Oscillator'] = (df['EMA 12'] - df['EMA 26']) / df['EMA 26'] 
        df.drop(labels=['EMA 12', 'EMA 26'], axis=1, inplace=True)


def add_Coefficient_of_Variation(dataframes: list):
    """Adds column containing the coefficient of variation indicator to each given dataframe.
    
    The coefficient of variation of price can be used to measure the volatility of a stock. 
    The indicator used here is calculated using the standard deviation of prices divided
    by a simple moving average of prices over a rolling window."""

    for df in dataframes:
        df['Std'] = df['Close'].rolling(window=10).std()
        df['SMA'] = df['Close'].rolling(window=10).mean()
        df.dropna(inplace=True)
        df['CV'] = df['Std'] / df['SMA'] 
        df.drop(labels=['Std', 'SMA'], axis=1, inplace=True)


def add_RSI(dataframes: list):
    """Adds two columns containing the RSI to the each given dataframe.
    The second will be scaled while the first will be kept as a benchmark.
    
    RSI (Relative Strength Index) is a simple momentum indicator used
    to identify overbought or oversold conditions. It is calculated using the 
    formula 100 - (100 / (1 + R)) where R is the mean gain divided by the mean loss over a rolling window.
    In practice, a value above 70 is typically used to indicate that an asset is overbought 
    and a value less tahn 30 is typically used to indicate that an asset is oversold."""

    for df in dataframes:
        df['Diff'] = df['Close'].diff()
        df.dropna(inplace=True)
        df['Mean gain'] = df['Diff'].rolling(window=10).apply(lambda x: x[x > 0].mean())
        df['Mean loss'] = df['Diff'].rolling(window=10).apply(lambda x: abs(x[x <= 0].mean()))
        df.dropna(inplace=True)
        df['RSI'] = 100 - (100 / (1 + (df['Mean gain'] / df['Mean loss'])))
        df['Scaled RSI'] = df['RSI']
        df.drop(labels=['Diff', 'Mean gain', 'Mean loss'], axis=1, inplace=True)


def add_pct_change(dataframes: list):
    """Adds column containing the percentage change from the previous 
    close price 5 steps previous to each given dataframe. This is used
    to capture short term momentum."""

    for df in dataframes:
        df['Pct Change'] = df['Close'].pct_change(periods=5)
        df.dropna(inplace=True)


def drop_volume(dataframes: list):
    """Drops the volume column from all given dataframes."""

    for df in dataframes:
        df.drop(labels=['Volume'], axis=1, inplace=True)


def reorder_columns(dataframes: list):
    """Reorders columns so that RSI column is the first indicator in the dataframe.
    This is just done to make indexing easier in the environment implementation."""

    columns_titles = ["Close", "RSI", "MACD", "MACD Percentage", "Volume Oscillator", "CV", "Scaled RSI", "Pct Change"]
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i][columns_titles]


def scale_data(dataframes: list):
    """ Scales the training data and uses these parameters for scaling the 
    validation and testing sets. The columns that will be used as the 
    state space for the reinforcement learning agent are then clipped
    to the range [-3, 3] to remove extreme outliers."""

    scaler = RobustScaler()
    columns = dataframes[0].columns
    scaled_training_data = scaler.fit_transform(dataframes[0].iloc[:, 3:])
    scaled_validation_data = scaler.transform(dataframes[1].iloc[:, 3:])
    scaled_testing_data = scaler.transform(dataframes[2].iloc[:, 3:])

    scaled_training_df = pd.DataFrame(scaled_training_data, columns=columns[3:])
    scaled_validation_df = pd.DataFrame(scaled_validation_data, columns = columns[3:])
    scaled_testing_df = pd.DataFrame(scaled_testing_data, columns = columns[3:])
    
    scaled_training_df.index = dataframes[0].index
    scaled_validation_df.index = dataframes[1].index
    scaled_testing_df.index = dataframes[2].index

    scaled_training_df = scaled_training_df.clip(lower=-3, upper=3)
    scaled_validation_df = scaled_validation_df.clip(lower=-3, upper=3)
    scaled_testing_df = scaled_testing_df.clip(lower=-3, upper=3)

    dataframes[0] = pd.concat([dataframes[0].iloc[:, :3], scaled_training_df], axis=1)
    dataframes[1] = pd.concat([dataframes[1].iloc[:, :3], scaled_validation_df], axis=1)
    dataframes[2] = pd.concat([dataframes[2].iloc[:, :3], scaled_testing_df], axis=1)




def visualize_data(dataframes: list):
    """Plots histograms of features in training, validation, and testing datasets
    after scaling and clipping. Used for quick visualization."""
    
    dataframes[0].iloc[:, 3:].hist(bins=15)
    plt.suptitle("Histograms of scaled and clipped features for training data")

    dataframes[1].iloc[:, 3:].hist(bins=15)
    plt.suptitle("Histograms of scaled and clipped features for validation data")

    dataframes[2].iloc[:, 3:].hist(bins=15)
    plt.suptitle("Histograms of scaled and clipped features for testing data")

    plt.show()



# example:
# python Data_Preprocessing.py "KO" "2005-01-01"

if __name__ == "__main__":

    stock_ticker = sys.argv[1]
    start_date = sys.argv[2]
    df = load_data(stock_ticker, start_date)
    dataframes = split_data(df)
    add_MACD(dataframes)
    add_Volume_Oscillator(dataframes)
    add_Coefficient_of_Variation(dataframes)
    add_RSI(dataframes)
    add_pct_change(dataframes)
    drop_volume(dataframes)
    reorder_columns(dataframes)
    scale_data(dataframes)
    print(dataframes[0].describe(), "\n")
    print(dataframes[1].describe(), "\n")
    print(dataframes[2].describe())
    visualize_data(dataframes)
    try:
        os.mkdir(f"./{stock_ticker}_scaled")
    except FileExistsError:
        print("Cannot create directory since it already exists, please delete it manually first if you wish to overwrite it.")
    dataframes[0].to_csv(f'./{stock_ticker}_scaled/Training.csv')
    dataframes[1].to_csv(f'./{stock_ticker}_scaled/Validation.csv')
    dataframes[2].to_csv(f'./{stock_ticker}_scaled/Testing.csv')