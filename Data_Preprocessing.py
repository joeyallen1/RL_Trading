import yfinance
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt



def load_data(stock_ticker: str, start_date: str) -> pd.DataFrame:
    """Returns a dataframe containing the close and volume data 
    for a given stock ticker and start date."""

    data = yfinance.download(stock_ticker, interval="1d", auto_adjust=True, start=start_date, multi_level_index=False)
    data = data.loc[:,['Close', 'Volume']]
    data.dropna(inplace=True)
    return data


def split_data(df: pd.DataFrame) -> tuple:
    """Splits the given dataframe into training, validation,
    and test sets."""

    length = len(df)
    training_df = df.iloc[:int(0.6 * length), :]
    validation_df = df.iloc[int(0.6*length):int(0.7*length), :]
    testing_df = df.iloc[int(0.7*length):, :]
    return training_df, validation_df, testing_df
    

def add_MACD(df: pd.DataFrame):
    """Adds column containing the MACD indicator and a column containing
    the MACD percentage indicator to the given dataframe.
    
    MACD (moving average convergence/divergence) is a momentum
    indicator that uses two exponential moving averages of a stock's 
    price. The regular MACD is calculated by subtracting the long term
    EMA from the short term EMA, and then the regular MACD is divided
    by the long term EMA to get the MACD percentage (in terms of the 
    long term EMA). The MACD percentage is then clipped to prevent values
    less than -1 and greater than 1. In practice, the regular MACD can be used
    as a buy signal when its value is greater than 0 and a sell
    signal when its value is less than 0."""

    df['EMA 3'] = df['Close'].ewm(span=3, adjust=False).mean()
    df['EMA 10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['MACD'] = df['EMA 3'] - df['EMA 10']
    df['MACD Percentage'] = np.clip((df['EMA 3'] - df['EMA 10']) / df['EMA 10'], -1.0, 1.0)
    df.drop(labels=['EMA 3' 'EMA 10'], axis=1, inplace=True)


def add_Volume_Oscillator(df: pd.DataFrame):
    """Adds column containing the volume oscillator indicator
    to the given dataframe.
    
    The volume oscillator captures trends in the trading volume of a stock.
    It is calculated by subtracting a long term moving average of the trading
    volume from a short term moving average. This value is then divided
    by the long term average to get a percentage value, which is 
    then clipped to the range [-1, 1]."""
    df['EMA 5'] = df['Volume'].ewm(span=5, adjust=False).mean()
    df['EMA 20'] = df['Volume'].ewm(span=20, adjust=False).mean()
    df['Volume Oscillator'] = np.clip((df['EMA 5'] - df['EMA 20']) / df['EMA 20'], -1.0, 1.0)
    df.drop(labels=['EMA 5' 'EMA 20'], axis=1, inplace=True)


def add_Coefficient_of_Variation(df: pd.DataFrame):
    """Adds column containing the coefficient of variation indicator to the given dataframe.
    
    The coefficient of variation of price can be used to measure the volatility of a stock. 
    The indicator used here is calculated using the standard deviation of prices divided
    by a simple moving average of prices over a rolling window. Here, the value is 
    clipped to the range [0, 1]."""

    df['Std'] = df['Close'].rolling(window=10).std()
    df['SMA'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    df['CV'] = np.clip(df['Std'] / df['SMA'], 0, 1)
    df.drop(labels=['Std', 'SMA'], axis=1, inplace=True)


def add_RSI(df: pd.DataFrame):
    """Adds column containing the RSI to the given dataframe.
    
    RSI (Relative Strength Index) is a simple momentum indicator used
    to identify overbought or oversold conditions. It is calculated using the 
    formula 100 - (100 / (1 + R)) where R is the mean gain divided by the mean loss over a rolling window.
    In practice, a value above 70 is typically used to indicate that an asset is overbought 
    and a value less tahn 30 is typically used to indicate that an asset is oversold.
    Here, the value is then divided by 100 to get an indicator in the range [0, 1]."""

    df['Diff'] = df['Close'].diff()
    df.dropna(inplace=True)
    df['Mean gain'] = df['Diff'].rolling(window=10).apply(lambda x: x[x > 0].mean())
    df['Mean loss'] = df['Diff'].rolling(window=10).apply(lambda x: abs(x[x <= 0].mean()))
    df.dropna(inplace=True)
    df['RSI'] = 100 - (100 / (1 + (df['Mean gain'] / df['Mean loss'])))
    df.drop(labels=['Diff', 'Mean gain', 'Mean loss'], axis=1, inplace=True)


def add_pct_change(df: pd.DataFrame):
    """Adds column containing the percentage change from the previous 
    close price to the given dataframe."""

    df['Pct Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    

def scale_and_save_data(dataframe, filename):
    df = dataframe.copy(deep=True)
    training_data = df.iloc[:int(df.shape[0] * 0.7), :].copy(deep=True)
    validation_data = df.iloc[int(df.shape[0] * 0.7):int(df.shape[0] * 0.8), :].copy(deep=True)
    testing_data = df.iloc[int(df.shape[0] * 0.8):, :].copy(deep=True)
    min_max_values = {}
    for label in df.columns:
        min_max_values[label] = [training_data[label].min(), training_data[label].max()]
        training_data[label] = (training_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
        validation_data[label] = (validation_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
        testing_data[label] = (testing_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
    df = pd.concat([training_data, validation_data, testing_data])
    df.to_csv(filename)


def visualize_data(df, stock_ticker, start_date):
    plt.subplot(2, 3, 1)
    plt.plot(df['Close'])
    plt.ylabel('Adjusted Close Price')
    plt.xlabel('Date')
    plt.title('Adjusted Close over Time')

    plt.subplot(2, 3, 2)
    plt.hist(df['Volume'], bins=15)
    plt.xlabel('Shares Traded')
    plt.ylabel('Frequency (Days)')
    plt.title('Distribution of Daily Trading Volume')

    plt.subplot(2, 3, 3)
    plt.hist(df['SMA Ratio'], bins=15)
    plt.xlabel('SMA Ratio')
    plt.ylabel('Frequency (Days)')
    plt.title('Distribution of Daily SMA Ratios')

    plt.subplot(2, 3, 4)
    plt.hist(df['RSI'], bins=15)
    plt.xlabel('RSI')
    plt.ylabel('Frequency (Days)')
    plt.title('Distribution of Daily RSI')

    plt.subplot(2, 3, 5)
    plt.hist(df['Bandwidth'], bins=15)
    plt.xlabel('Bollinger Band Width')
    plt.ylabel('Frequency (Days)')
    plt.title('Distribution of Bollinger Band Widths')

    plt.subplot(2, 3, 6)
    plt.plot(df['Pct Change'])
    plt.ylabel('Percent Change from previous close price')
    plt.xlabel('Date')
    plt.title('Percent Change in Price over time')

    plt.suptitle(f"Dataset Visualization of {stock_ticker} stock starting on {start_date} (start date may be clipped)")
    plt.tight_layout()
    plt.show()


# used KO and start date of 2005-01-01
# have to close out of matplotlib window in order to save data 
# (keyboard interrupt doesn't work)
if __name__ == "__main__":
    stock_ticker = sys.argv[1]
    start_date = sys.argv[2]
    df = load_data(stock_ticker, start_date)
    training_df, validation_df, testing_df = split_data(df)

    # df = add_SMA(df)
    # df = add_RSI(df)
    # df = add_bandwidth(df)
    # df = add_pct_change(df)
    visualize_data(df, stock_ticker, start_date)
    scale_and_save_data(df, f"{stock_ticker}_data.csv")


    # remember to drop NaN or 0 values 