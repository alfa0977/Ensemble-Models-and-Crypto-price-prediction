import numpy as np
import pandas as pd
import os
import streamlit as st

import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.header("Week 11: ensemble learning".upper())
st.subheader("a review on ensemble learning tools".capitalize())
st.markdown("First page will show you how each model is structured. " \
"Next pages would show examples of running each model and their details.")

st.divider()
tex="## 🧠 Project Overview\n\
**Problem:** Forecasting the daily BTC/USDT closing price using recent daily candlestick data (Date, Open, High, Low, Close).\n\n \
**Goal:** Build, tune, and compare ensemble regressors to minimize test error and produce a reliable next‑day closing price forecast.\n\n \
**Approach:** Fetch Binance daily data via API. Then tune RandomForest, AdaBoost, and XGBoost with GridSearchCV; construct Stacking \
      (Ridge meta‑model) and Voting ensembles, then compute MAPE per model."
st.markdown(tex)