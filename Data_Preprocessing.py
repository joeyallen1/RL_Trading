import yfinance
import pandas as pd
import numpy as np
import sys

def load_data(stock_ticker, start_date):
    data = yfinance.download(stock_ticker, interval="1d", auto_adjust=True, start=start_date)
    data = data.loc[:,['Close', 'Volume']]
    data.dropna(inplace=True)
    return data

def add_SMA(dataframe):
    df = dataframe.copy(deep=True)
    df['SMA 10'] = df.rolling(window=10)['Close'].mean()
    df['SMA 50'] = df.rolling(window=50)['Close'].mean()
    df.dropna(inplace=True)
    df['SMA Ratio'] = df['SMA 10'] / df['SMA 50']
    df.drop(labels=['SMA 10', 'SMA 50'], axis=1, inplace=True)
    return df

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

def add_bandwidth(dataframe):
    df = dataframe.copy(deep=True)
    df['Std dev'] = df['Close'].rolling(window=20).std()
    df.dropna(inplace=True)
    df['SMA 20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    df['Bandwidth'] = ((df['SMA 20'] + (2 * df['Std dev'])) - (df['SMA 20'] - (2 * df['Std dev']))) / df['SMA 20']
    df.drop(labels=['Std dev', 'SMA 20'], axis=1, inplace=True)
    return df

def split_and_save_data(dataframe, filename):
    df = dataframe.copy(deep=True)
    training_data = df.iloc[:int(df.shape[0] * 0.7), :].copy(deep=True)
    validation_data = df.iloc[int(df.shape[0] * 0.7):int(df.shape[0] * 0.8), :].copy(deep=True)
    testing_data = df.iloc[int(df.shape[0] * 0.8):, :].copy(deep=True)
    min_max_values = {}
    for label in ['Close', 'Volume', 'SMA Ratio', 'RSI', 'Bandwidth']:
        min_max_values[label] = [training_data[label].min(), training_data[label].max()]
        training_data[label] = (training_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
        validation_data[label] = (validation_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
        testing_data[label] = (testing_data[label] - min_max_values[label][0]) / (min_max_values[label][1] - min_max_values[label][0])
    df = pd.concat([training_data, validation_data, testing_data])
    df.to_csv(filename)

if __name__ == "__main__":
    stock_ticker = sys.argv[0]
    start_date = sys.argv[1]
    df = load_data(stock_ticker, start_date)
    df = add_SMA(df)
    df = add_RSI(df)
    df = add_bandwidth(df)
    split_and_save_data(df, f"{stock_ticker}_data.csv")