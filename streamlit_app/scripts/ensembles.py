import requests  # For making HTTP requests to APIs
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime, timedelta  # For handling dates and times
import warnings  # For managing warning messages
import streamlit as st
from sklearn.model_selection import train_test_split  # For splitting data

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


def compute_mape(y_true, y_pred):
    # Guard against division by zero errors
    y_true = np.array(y_true).flatten()  # Flatten true values
    y_pred = np.array(y_pred).flatten()  # Flatten predicted values

    # Handle zero values
    valid_indices = y_true != 0  # Indices where true value is not zero
    if valid_indices.any():
        y_true = y_true[valid_indices]  # Filter true values
        y_pred = y_pred[valid_indices]  # Filter predicted values

    # Compute MAPE
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))  # Calculate MAPE
    return mape  # Return MAPE