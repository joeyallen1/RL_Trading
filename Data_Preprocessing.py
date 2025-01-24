import yfinance
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

df = pd.DataFrame()
df.rolling(window=3).


def load_data(stock_ticker, start_date):
    data = yfinance.download(stock_ticker, interval="1d", auto_adjust=True, start=start_date, multi_level_index=False)
    data = data.loc[:,['Close', 'Volume']]
    data.dropna(inplace=True)
    return data


# SMA Ratio:
# - simple trend indicator
# - uses a short term and longer term simple moving average
# - a ratio value above 1 typically indicates a bullish signal and below 1 indicates a bearish signal
# - calculated here using 10 and 50 day windows
# def add_SMA(dataframe):
#     df = dataframe.copy(deep=True)
#     df['SMA 10'] = df.rolling(window=10)['Close'].mean()
#     df['SMA 50'] = df.rolling(window=50)['Close'].mean()
#     df.dropna(inplace=True)
#     df['SMA Ratio'] = df['SMA 10'] / df['SMA 50']
#     df.drop(labels=['SMA 10', 'SMA 50'], axis=1, inplace=True)
#     return df


def add_MACD(dataframe):
    df = dataframe.copy(deep=True)
    df['EMA 3'] = df['Close'].ewm(span=3, adjust=False).mean()
    df['EMA 10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['MACD'] = df['EMA 10'] = df['EMA 3']



# RSI (Relative Strength Index):
# - simple momentum indicator, used to identify overbought or oversold conditions
# - calculated using formula 100 - (100 / (1 + RSI)) where RSI is the mean gain divided by the mean loss over some time period
# - a value above 70 is typically used to indicate that an asset is overbought and a value less tahn 30 is typically used to indicate that an asset is oversold
# - calculated here using a 14 day rolling window
def add_RSI(dataframe):
    df = dataframe.copy(deep=True)
    df['Diff'] = df['Close'].diff()
    df.dropna(inplace=True)
    df['Mean gain'] = df['Diff'].rolling(window=14).apply(lambda x: x[x > 0].mean())
    df['Mean loss'] = df['Diff'].rolling(window=14).apply(lambda x: abs(x[x <= 0].mean()))
    df.dropna(inplace=True)
    df['RSI'] = 100 - (100 / (1 + (df['Mean gain'] / df['Mean loss'])))
    df.drop(labels=['Diff', 'Mean gain', 'Mean loss'], axis=1, inplace=True)
    return df


# Bollinger Bands:
# - used to measure price volatility of an asset 
# - calculated with two bands, two standard deviations above and below a moving average line
# - the distance between the bands indicates volatility of the asset
# - also used to indicate overbought or oversold conditions - overbought when the price moves above the upper band and oversold when the price moves below the lower band
# - here, just the Bollinger Bandwidth is used as a volatility indicator for simplicity and reducing the number of redundant features- a large bandwidth indicates high volatility and smaller bandwidth indicates lower volatility 
# - the bandwidth is calculated as: (upper band - lower band)/ middle band for a given window (20 periods in this case)
def add_bandwidth(dataframe):
    df = dataframe.copy(deep=True)
    df['Std dev'] = df['Close'].rolling(window=20).std()
    df.dropna(inplace=True)
    df['SMA 20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    df['Bandwidth'] = ((df['SMA 20'] + (2 * df['Std dev'])) - (df['SMA 20'] - (2 * df['Std dev']))) / df['SMA 20']
    df.drop(labels=['Std dev', 'SMA 20'], axis=1, inplace=True)
    return df


def add_pct_change(dataframe):
    df = dataframe.copy(deep=True)
    df['Pct Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

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
    df = add_SMA(df)
    df = add_RSI(df)
    df = add_bandwidth(df)
    df = add_pct_change(df)
    visualize_data(df, stock_ticker, start_date)
    scale_and_save_data(df, f"{stock_ticker}_data.csv")