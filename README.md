# Ensembles-in-Machine-Learning

## Overview

This project addresses the challenging task of cryptocurrency price prediction, focusing specifically on Bitcoin (BTC-USD) price forecasting. In financial markets characterized by high volatility and non-linear patterns, traditional forecasting methods often fall short of providing accurate predictions.

Ensemble learning methods offer a powerful solution to this problem by combining multiple machine learning models to achieve better predictive performance than any individual model could accomplish alone. By leveraging the collective intelligence of various algorithms, we can capture different aspects of the complex price patterns in cryptocurrency markets.

This project implements and compares several state-of-the-art ensemble methods:

1. **Random Forest** - A bagging-based ensemble that builds multiple decision trees and merges their predictions
2. **AdaBoost** - A boosting algorithm that sequentially builds models by focusing on previously misclassified instances
3. **XGBoost** - An optimized gradient boosting framework known for its speed and performance
4. **Stacking** - A meta-learning approach that uses predictions from multiple models as inputs to a final model
5. **Voting** - A simple but effective method that combines predictions through weighted or unweighted averaging

The implementation includes rigorous hyperparameter tuning, feature importance analysis, and model evaluation using appropriate metrics like Mean Absolute Percentage Error (MAPE). By visualizing ensemble architectures and comparing model performance, this project provides both educational value about ensemble methods and practical utility for cryptocurrency price prediction.

Beyond the predictive models themselves, this repository serves as a comprehensive tutorial on implementing ensemble learning techniques in Python, with visualizations that explain the core concepts behind different ensemble approaches.

## 📁 Project Structure

```
Ensembles-in-Machine-Learning/
│
├── btc_usd.py                          # Main script for Bitcoin price prediction using ensemble models
│                                       # Implements data fetching, preprocessing, model training and evaluation
│
├── ensembled_plots.py                  # Visualization script for ensemble learning concepts
│                                       # Creates network diagrams to explain different ensemble methods
│
├── output/                             # Generated visualizations and results
│   ├── bagging_plot.png                # Visual representation of the bagging ensemble method
│   ├── boosting_plot.png               # Visual representation of the boosting ensemble method
│   ├── stacking_plot.png               # Visual representation of the stacking ensemble method
│   └── voting_plot.png                 # Visual representation of the voting ensemble method
│
├── BTC_USD.ipynb                       # Jupyter notebook version of the Bitcoin price prediction model
│
├── Ensembled_plots.ipynb               # Jupyter notebook version of the ensemble concept visualizations
│
├── requirements.txt                    # List of Python package dependencies with versions
│                                       # Includes scikit-learn, xgboost, yfinance, matplotlib, etc.
│
├── .gitignore                          # Specifies files and directories to ignore in git versioning
│                                       # Excludes venv/, __pycache__/, output/, and other generated files
│
│
└── README.md                           # Project documentation and overview
```

## 🔧 Setup and Installation Instructions

For detailed instructions please refer to the [Instructions](./Instructions.md) file.