import requests  # For making HTTP requests to APIs
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime, timedelta  # For handling dates and times
import warnings  # For managing warning messages
import streamlit as st
from sklearn.model_selection import train_test_split  # For splitting data

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Define the Binance API endpoint for daily BTC/USDT candlestick data
#key = "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime=1735689600000"
# key="https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
# req={"vs_currency":"usd","days":"365"}
key="https://api.huobi.pro/market/history/kline"
req={"symbol":"btcusdt","period":"1day","size":"365"}
# Send a GET request to the API endpoint
data = requests.get(key,req)
data = data.json()  # Parse the response as JSON
# Extract relevant price information from the API response
# prices = [
#     {
#         "Date": item[0],  # Timestamp in milliseconds
#         "Open": float(item[1]),  # Opening price for the day
#         "High": float(item[2]),  # Highest price for the day
#         "Low": float(item[3]),  # Lowest price for the day
#         "Close": float(item[4]),  # Closing price for the day
#     }
#     for item in data
# ]
prices =  data["data"]


# Convert the list of price dictionaries into a pandas DataFrame
btc_usdt = pd.DataFrame(prices)
btc_usdt['id']=btc_usdt['id']*1e9
btc_usdt["id"]=pd.to_datetime(btc_usdt["id"],origin='unix')
btc_usdt.set_index("id", inplace=True)  # Set 'Date' as the index
btc_usdt.sort_index(ascending=True, inplace=True)
st.dataframe(btc_usdt)
st.info(f"Loaded data shape: {btc_usdt.shape}")
#st.dataframe(btc_usdt.head())

st.divider()
# Ensure we have all the necessary columns and handle any missing values
btc_usdt = btc_usdt.fillna(method="ffill")  # Forward fill missing values

st.write("\nColumns in dataset:", list(btc_usdt.columns))  # Print column names
btc_usdt=btc_usdt.drop(["amount","vol","count"],axis=1)
# Splitting the dataset - use all available features to predict 'Close' price
X = btc_usdt.drop("close", axis=1)  # Features: all columns except 'Close'
y = btc_usdt["close"]  # Target: 'Close' price
with st.expander("Features and targets"):
    col1,col2=st.columns([0.7,0.3])
    with col1:
        st.write("features".upper())
        st.dataframe(X)
    with col2:
        st.write("targets".upper())
        st.dataframe(y)
st.write(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")  # Print shapes

# Time-based split for financial data (no shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
st.header("Different Models Comparison")
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs(["Random Forest", "AdaBoost","XGBoost",
                                            "Gradient Boosting","Hist Gradient Boosting",
                                            "Stacking","Voting"])
with tab1:
    ct_RF=st.container(border=True) #container for Random forest
with tab2:
    ct_AdaB=st.container(border=True) #container for AdaBoost
with tab3:
    ct_XGB=st.container(border=True) #container for XGboost
with tab4:
    ct_GB=st.container(border=True)
with tab5:
    ct_HGB=st.container(border=True)
with tab6:
    ct_stck=st.container(border=True) #container for stacking
with tab7:
    ct_vt=st.container(border=True) #container for voting
ct_RF.header("Random Forest")
ct_AdaB.header("AdaBoost")
ct_XGB.header("XGBoost")
ct_GB.header("Gradient Boosting")
ct_HGB.header("Hist Gradient Boosting")
ct_stck.header("Stacking")
ct_vt.header("Voting")

from scripts.visualization import (plot_feature_importance,
                                   plot_predictions)
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning

# **************************************************
#           Random Forest
# **************************************************
from sklearn.ensemble import RandomForestRegressor  # Random Forest model


# Hyperparameters to tune for Random Forest
param_grid_rf = {
    "n_estimators": [50, 100, 150],  # Number of trees
    "max_features": ["auto", "sqrt"],  # Number of features to consider
    "max_depth": [10, 20, 30, None],  # Maximum depth of trees
    "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4],  # Minimum samples at a leaf node
}

grid_search_rf = GridSearchCV(
    RandomForestRegressor(),  # Model to tune
    param_grid_rf,  # Parameter grid
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=2,  # Verbosity level
    scoring="neg_mean_squared_error",  # Scoring metric
)
with ct_RF:
    with st.spinner(text="Grid Search for Random Forest"):
        grid_search_rf.fit(X_train, y_train)  # Fit grid search to training data
        best_rf = grid_search_rf.best_estimator_  # Get best Random Forest model
        rf_pred_tuned = best_rf.predict(X_test)  # Random Forest predictions
ct_RF.pyplot(plot_predictions("Random Forest", rf_pred_tuned,X_test,y_test))
# **************************************************
#                       AdaBoost
# **************************************************
from sklearn.ensemble import AdaBoostRegressor  # AdaBoost model
param_grid_ada = {
    "n_estimators": [50, 100, 150],  # Number of boosting rounds
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],  # Learning rate
}

grid_search_ada = GridSearchCV(
    AdaBoostRegressor(),  # Model to tune
    param_grid_ada,  # Parameter grid
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=2,  # Verbosity level
    scoring="neg_mean_squared_error",  # Scoring metric
)
with ct_AdaB:
    with st.spinner(text="Grid Search for AdaBoost"):
        grid_search_ada.fit(X_train, y_train)  # Fit grid search to training data
        best_ada = grid_search_ada.best_estimator_  # Get best AdaBoost model
        ada_pred_tuned = best_ada.predict(X_test)  # AdaBoost predictions
ct_AdaB.pyplot(plot_predictions("AdaBoost", ada_pred_tuned,X_test,y_test))

# **************************************************
#                       XGBoost
# **************************************************
import xgboost as xgb  # XGBoost library

param_grid_xgb = {
    "learning_rate": [0.01, 0.05, 0.1],  # Learning rate
    "n_estimators": [100, 150, 200],  # Number of boosting rounds
    "max_depth": [3, 5, 7, 10],  # Maximum tree depth
    "subsample": [0.8, 0.9, 1],  # Subsample ratio
    "colsample_bytree": [0.8, 0.9, 1],  # Feature subsample ratio
}

grid_search_xgb = GridSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror"),  # Model to tune
    param_grid_xgb,  # Parameter grid
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=2,  # Verbosity level
    scoring="neg_mean_squared_error",  # Scoring metric
)
with ct_XGB:
    with st.spinner("Grid Search XGBoost"):
        grid_search_xgb.fit(X_train, y_train)  # Fit grid search to training data
        best_xgb = grid_search_xgb.best_estimator_  # Get best XGBoost model
        xgb_pred_tuned = best_xgb.predict(X_test)  # XGBoost predictions

ct_XGB.pyplot(plot_predictions("XGBoost", xgb_pred_tuned,X_test,y_test))
#st.divider()
# **********************************************************************
#                     Gradient Boosting
# **********************************************************************
from sklearn.ensemble import GradientBoostingRegressor
param_grid_gb={
    "n_estimators":[50,100,150], # number of boosting rounds
    "learning_rate":[0.01,0.05,0.1,0.5,1], #learning rate
}
grid_search_gb=GridSearchCV(
    GradientBoostingRegressor(),
    param_grid_gb,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="neg_mean_squared_error"
)
with ct_GB:
    with st.spinner(text="Grid Search for Gradient Boosting"):
        grid_search_gb.fit(X_train,y_train)
        best_gb=grid_search_gb.best_estimator_
        gb_pred_tuned=best_gb.predict(X_test)
ct_GB.pyplot(plot_predictions("Gradient Boosting",gb_pred_tuned,X_test,y_test))
# *************************************************************
#                 Hist Gradient Boosting
# *************************************************************
from sklearn.ensemble import HistGradientBoostingRegressor
param_grid_hgb={
    #"n_estimators":[50,100,150],
    "learning_rate":[0.01,0.05,0.1,0.5,1]
}
HistGradientBoostingRegressor()
grid_search_hgb=GridSearchCV(
    HistGradientBoostingRegressor(),
    param_grid_hgb,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="neg_mean_squared_error"
)
with ct_HGB:
    with st.spinner(text="Grid Search for Hist Gradient Boosting"):
        grid_search_hgb.fit(X_train,y_train)
        best_hgb=grid_search_hgb.best_estimator_
        hgb_pred_tuned=best_hgb.predict(X_test)
        st.pyplot(plot_predictions("Hist Gradient Boosting",hgb_pred_tuned,X_test,y_test))
   

# **************************************************
#                       Stacking
# **************************************************
from sklearn.ensemble import StackingRegressor  # Stacking ensemble
from sklearn.linear_model import Ridge  # Ridge regression for meta-learner

base_learners = [("rf", best_rf), ("ada", best_ada), ("xgb", best_xgb)]  # Base models

param_grid_stack = {
    "final_estimator__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}  # Ridge alpha

stack_reg = StackingRegressor(
    estimators=base_learners, final_estimator=Ridge()
)  # Stacking model
grid_search_stack = GridSearchCV(
    stack_reg,  # Model to tune
    param_grid_stack,  # Parameter grid
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=2,  # Verbosity level
    scoring="neg_mean_squared_error",  # Scoring metric
)
with ct_stck:
    with st.spinner(text="Grid Search for Stacking"):
        grid_search_stack.fit(X_train, y_train)  # Fit grid search to training data
        best_stack = grid_search_stack.best_estimator_  # Get best stacking model
        stack_pred_tuned = best_stack.predict(X_test)  # Stacking predictions
ct_stck.pyplot(plot_predictions("Stacking", stack_pred_tuned,X_test,y_test))
# st.divider()
# ct_vt.header("Voting")
# **************************************************
#                       Voting
# **************************************************
from sklearn.ensemble import VotingRegressor  # Voting ensemble

voting_reg = VotingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(n_estimators=100)),  # Random Forest
        ("ada", AdaBoostRegressor(n_estimators=100)),  # AdaBoost
        (
            "xgb",
            xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100),
        ),  # XGBoost
    ]
)
with ct_vt:
    with st.spinner("Grid Search for Voting"):
        voting_reg.fit(X_train, y_train)  # Fit voting regressor to training data
        vote_pred = voting_reg.predict(X_test)  # Predict on test data
ct_vt.pyplot(plot_predictions("Voting Regressor", vote_pred,X_test,y_test))
# with st.container():
#     with st.spinner("predicting using best models"):
        # Making predictions with each of the models
        #rf_pred_tuned = best_rf.predict(X_test)  # Random Forest predictions
        #ada_pred_tuned = best_ada.predict(X_test)  # AdaBoost predictions
        #xgb_pred_tuned = best_xgb.predict(X_test)  # XGBoost predictions
        #stack_pred_tuned = best_stack.predict(X_test)  # Stacking predictions

# Define a function to plot predictions
#import matplotlib.pyplot as plt  # Import plotting library


# Use the function to plot each model's results
#ct_RF.pyplot(plot_predictions("Random Forest", rf_pred_tuned,X_test,y_test))
#ct_AdaB.pyplot(plot_predictions("AdaBoost", ada_pred_tuned,X_test,y_test))
#ct_XGB.pyplot(plot_predictions("XGBoost", xgb_pred_tuned,X_test,y_test))
#ct_stck.pyplot(plot_predictions("Stacking", stack_pred_tuned,X_test,y_test))
#ct_vt.pyplot(plot_predictions("Voting Regressor", vote_pred,X_test,y_test))

# with st.container(border=True):
#     tab1,tab2,tab3,tab4=st.tabs(["Gradient Boosting","Hist Gradient Boosting",
#                                  "LightGBM","CatBoost",])
# with st.tab1:
#     ct_GB=st.container(border=True)
# with st.tab2:
#     ct_HGB=st.container(border=True)
# with st.tab3:
#     ct_LGB=st.container(border=True)
# with st.tab4:
#     ct_CB=st.container(border=True)

 
# *************************************************************
#                   LightGBM
# *************************************************************
#from lightgbm

st.pyplot(plot_feature_importance(
    best_rf.feature_importances_, X_train.columns, "Random Forest"
))  # RF feature importance
st.pyplot(plot_feature_importance(
    best_ada.feature_importances_,X_train.columns, "AdaBoost"
)) #adaboost feature importance
st.pyplot(plot_feature_importance(
    best_xgb.feature_importances_, X_train.columns, "XGBoost"
))  # XGB feature importance


st.pyplot(plot_feature_importance(
    best_gb.feature_importances_,X_train.columns, "Gradient Boosting"))


import numpy as np  # Import numpy
import matplotlib.pyplot as plt  # Import plotting library
from scripts.ensembles import compute_mape

# Calculate MAPE for each model
mape_rf = compute_mape(y_test, rf_pred_tuned)  # Random Forest MAPE
mape_ada = compute_mape(y_test, ada_pred_tuned)  # AdaBoost MAPE
mape_xgb = compute_mape(y_test, xgb_pred_tuned)  # XGBoost MAPE
mape_gb  = compute_mape(y_test,gb_pred_tuned)
mape_hgb = compute_mape(y_test,xgb_pred_tuned)
mape_stack = compute_mape(y_test, stack_pred_tuned)  # Stacking MAPE
mape_vote = compute_mape(y_test, vote_pred)  # Voting Regressor MAPE

# Visualization
models = [
    "Random Forest",
    "AdaBoost",
    "XGBoost",
    "Gradient Boosting",
    "Hist Gradient Boosting",
    "Stacking",
    "Voting Regressor",
]  # Model names
mape_values = [mape_rf, mape_ada, mape_xgb,mape_gb,mape_hgb, mape_stack, mape_vote]  # MAPE values

fig=plt.figure(figsize=(10, 6))  # Set figure size
plt.bar(
    models, mape_values, color=["blue", "green", "red","cyan" , "yellow", "purple", "orange"]
)  # Bar plot
plt.ylabel("MAPE (%)")  # Y-axis label
plt.title("Mean Absolute Percentage Error (MAPE) for Different Models")  # Title
plt.xticks(rotation=45)  # Rotate x-tick labels
plt.tight_layout()  # Adjust layout
plt.savefig("mape_comparison.png")  # Save plot to file
# plt.show()  # Display plot
st.pyplot(fig)

# Print out the computed MAPE values
with st.container(border=True):
    st.write(f"Random Forest MAPE:\t\t {mape_rf:.2f}%")
    st.write(f"AdaBoost MAPE:\t\t {mape_ada:.2f}%")
    st.write(f"XGBoost MAPE:\t\t {mape_xgb:.2f}%")
    st.write(f"Gradient Boosting MAPE:\t\t {mape_xgb:.2f}%")
    st.write(f"Hist Gradient Boosting MAPE:\t\t {mape_xgb:.2f}%")
    st.write(f"Stacking MAPE:\t\t {mape_stack:.2f}%")
    st.write(f"Voting Regressor MAPE:\t\t {mape_vote:.2f}%")

# Make prediction for next day
latest_date = btc_usdt.index[-1]  # Get last date in dataset
next_day_date = (latest_date + timedelta(days=1)).strftime(
    "%Y-%m-%d"
)  # Next day's date

# Use the last available data as input for prediction
next_day_data = btc_usdt.iloc[-1:].drop(columns=["close"])  # Last row, drop 'Close'

# Predict using the best model
next_day_pred_rf  = best_rf.predict(next_day_data)  # RF prediction
next_day_pred_xgb = best_xgb.predict(next_day_data)  # XGB prediction
next_day_pred_gb  = best_gb.predict(next_day_data)
next_day_pred_hgb = best_hgb.predict(next_day_data)
next_day_pred_stack = best_stack.predict(next_day_data)  # Stacking prediction

with st.container(border=True):
    st.title(f"\nPredictions for {next_day_date}:")
    st.write(f"Random Forest BTC-USD Closing:\t\t ${next_day_pred_rf[0]:.2f}")
    st.write(f"XGBoost BTC-USD Closing:\t\t ${next_day_pred_xgb[0]:.2f}")
    st.write(f"Gradient Boosting BTC-USD Closing:\t\t ${next_day_pred_gb[0]:.2f}")
    st.write(f"Hist Gradient Boosting BTC-USD Closing:\t\t ${next_day_pred_hgb[0]:.2f}")
    st.write(f"Stacking Ensemble BTC-USD Closing:\t\t ${next_day_pred_stack[0]:.2f}")
    st.write(f"True prediction BTC-USD Closing:\t\t ${btc_usdt['close'].iloc[-1]:.2f}")