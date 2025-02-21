# Reinforcement Learning for Stock Trading

A trading bot for single-stock trading trained using reinforcement learning (RL).

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Approach](#approach)
4. [Limitations](#limitations)
5. [Current Work](#current-work)
6. [Future Steps](#future-steps)

## Overview
This project explores the application of reinforcement learning (RL) to single-stock trading. The goal is to develop an agent that learns an optimal trading strategy by interacting with market data and maximizing profit over a given period. The agent manages its budget by deciding daily whether to buy, sell, or hold a stock based on historical price and volume data. 

The project is implemented using Stable-Baselines3's Deep Q-Network (DQN) algorithm, a reinforcement learning technique that combines deep learning with value-based decision-making. The agent's performance is evaluated against baseline strategies such as buy-and-hold and simple indicator-based strategies.

## Features
- **Historical Data Processing:** Stock price and volume data sourced from Yahoo Finance
- **Feature Engineering:** Technical indicators extracted to enhance decision-making
- **Custom Gym Environment:** Defined for training RL agents using Stable-Baselines3
- **Discrete Action Space:** Buy, sell, and hold actions implemented within a structured RL setting
- **Training & Evaluation:** Model performance assessed on separate validation and test datasets
- **Hyperparameter Tuning:** Adjusting learning parameters for improved training and generalization
- **Unit Testing:** Environment logic verified to ensure stability and correctness

## Approach
Reinforcement learning is a framework where an agent interacts with an environment and learns an optimal policy by receiving rewards for its actions. Unlike traditional supervised learning, where labeled data dictates decisions, RL allows the agent to explore and optimize over time.

### Methods & Libraries
- **Data Handling:** Yahoo Finance API for stock data retrieval, Pandas for data manipulation
- **Feature Engineering:** Pandas and NumPy for computing technical indicators such as moving averages and RSI
- **Reinforcement Learning Framework:** OpenAI Gym for environment design, Stable-Baselines3 for implementing DQN
- **Performance Evaluation:** Matplotlib for visualization, backtesting against baseline strategies
- **Hyperparameter Tuning:** Optuna for tuning training parameters
- **Unit Testing:** Pytest for testing of environment logic

### Environment Design
- **State Representation:** Each day's state consists of technical indicators derived from historical stock data.
- **Actions:** The agent can buy, sell, or hold the stock.
- **Reward Function:** Profit is measured relative to a buy-and-hold strategy.
- **Data Representation:** Stock data is structured as a pandas DataFrame, where each row corresponds to a trading day.
- **Training Algorithm:** DQN is used to approximate the optimal Q-values for decision-making.

The agent is trained on a subset of the data, validated on a separate set for hyperparameter tuning, and finally tested on unseen data.

## Limitations
While reinforcement learning offers potential advantages in financial markets, several challenges exist:
- **Market Non-Stationarity:** Stock market dynamics change over time, making it difficult for an RL model to generalize.
- **External Influences:** Market movements are driven by factors beyond historical price data, such as news, sentiment, and economic events.
- **Feature Lagging:** Many technical indicators rely on past data and may not provide predictive insights.
- **Algorithm Constraints:** DQN operates in a discrete action space and is sensitive to hyperparameter tuning.

## Current Work
The trained model is currently being refined and benchmarked against simple indicator-based strategies to evaluate its effectiveness.

## Future Steps
- **Enhanced Feature Engineering:** Incorporate sentiment analysis and macroeconomic indicators.
- **Continuous Action Spaces:** Explore algorithms like PPO and actor-critic methods for more granular trade execution.
- **Portfolio Optimization:** Expand from single-stock trading to multi-asset allocation strategies.

This project serves as a foundation for exploring reinforcement learning in finance and highlights both the potential and challenges of applying RL techniques to real-world trading problems.