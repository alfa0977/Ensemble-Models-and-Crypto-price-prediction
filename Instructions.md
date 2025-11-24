# 📚 Instructions to run: Ensembeles Models in Bitcoin

Welcome to the Ensembeles Models in Bitcoin project! This comprehensive guide will walk you through every step of building a complete machine learning solution from acquiring data, preprocessing, train ensembles models to evaluation to showcasing your work and building your online presence, **needed to get hired in the industry**.

## 🎯 Learning Objectives

By completing this project, you will learn:
- 🧹 **Data Cleaning & Feature Engineering:** Learn to preprocess and transform raw Bitcoin data into features suitable for ensemble modeling.
- 🧠 **Ensemble Machine Learning:** Build, train, and compare multiple ensemble models (e.g., Random Forest, XGBoost) for Bitcoin price prediction.
- 📊 **Model Evaluation:** Analyze and interpret model performance using appropriate metrics for regression tasks and choose the best performance model.


## 🚀 Quick Start

Before diving into the full project, we recommend starting with our simplified Party-Time jupyter notebook in Google Colab. This **condensed version** introduces the **main concepts** and workflow without the complexity of the complete implementation. 
Once you're comfortable with the fundamentals, **return here for the comprehensive walkthrough**.

**📓 [Access Party-Time Notebook](https://colab.research.google.com/drive/1teaxxMv6Dkm6QdAAfdfwTndD-WQl7IkR)** - *A beginner-friendly introduction to get you started*




## 🚀 Getting Started

Follow this comprehensive step-by-step workflow to complete the project. Each step includes both execution instructions and understanding of what you're accomplishing.

### Step 1: Environment Setup

**Step 1: Fork the Repository**

You already have the code, so you can just upload it to your own GitHub account.

```bash
# Clone the repository
git clone <url-to-your-forked-repo-from-steps-above>
cd Week_11_Ensemble_Models

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate ensemble-bitcoin

# (Optional) Update the environment if you change dependencies
conda env update -f environment.yml --prune

# Add conda kernel to jupyter notebook
conda install ipykernel
python -m ipykernel install --user --name ensemble-bitcoin --display-name "ensemble-bitcoin"
```

**Additional Setup Steps:**
* **VSCode Python Interpreter Setup:**
   - **Windows/Linux:** Press `Ctrl+Shift+P` 
   - **Mac:** Press `Cmd+Shift+P` 
- Select "Python: Select Interpreter", then choose the "ensemble-bitcoin" interpreter.
* Once you open the Jupyter Notebook, it should automatically use the "ensemble-bitcoin" kernel. If not, please restart VSCode. And if not successful, then on the top right corner of the notebook, you can manually select the kernel by clicking on it and choosing "ensemble-bitcoin". You most likely will find it in the Jupyter kernel list.


🌟 **Alternative: Python Virtual Environment Setup**  
Instead of using Conda, you can opt for a Python virtual environment. It's lightweight and more production-friendly, though you might encounter dependency conflicts. For detailed instructions, refer to the [📄 Setup Environment Guide](Docs/3.Setup_Environment.md).  


### Step 2: Run the ensemble visualization script

```bash
python ensembled_plots.py
```

This script creates visual network diagrams that explain the architecture and data flow of four key ensemble learning methods:
- **Bagging**: Shows how multiple base models are trained on bootstrapped samples and combined through averaging or voting
- **Boosting**: Illustrates the sequential learning process where each model corrects errors from previous models
- **Stacking**: Demonstrates how predictions from multiple models become inputs for a meta-learner
- **Voting**: Presents how different models' predictions are combined through weighted or unweighted averaging

After execution, four PNG images will be generated in the project root directory.

There is a Juypter notebook version of the ensemble visualization script that allows for interactive exploration of the visualization parameters and explanations.

### Step 3: Run the Bitcoin price prediction script

```bash
python btc_usd.py
```

This is the main script that implements a complete Bitcoin price prediction workflow:
1. Downloads historical BTC-USD price data using yfinance
2. Preprocesses data and creates technical indicator features
3. Splits data into training and testing sets
4. Trains multiple ensemble models with hyperparameter tuning:
   - Random Forest
   - AdaBoost
   - XGBoost
   - Stacking Regressor
   - Voting Regressor
5. Evaluates models using Mean Absolute Percentage Error (MAPE)
6. Generates feature importance plots for tree-based models
7. Makes next-day BTC-USD price predictions

The script outputs performance metrics, plots, and predictions directly in the console.


## 🔄 Potential Changes

Explore these ideas to further enhance your project:

1. **Additional Ensemble Techniques**: Experiment with other ensemble methods such as Gradient Boosting, LightGBM, or CatBoost for potentially better performance.
2. **Model Interpretability**: Apply SHAP or LIME to explain predictions and feature importance.
3. **Deployment**: Package your best model into a Streamlit or Flask web app for interactive predictions.



### 🏆 Change the Project and Showcase Your Skills on GitHub

Follow these steps to make your own improvements to the project and demonstrate your learning:

1. **Create a New Branch for Your Work**
   - It's best practice to make changes on a new branch:
     ```bash
     git checkout -b my-feature-branch
     ```

2. **Make Your Changes**
   - Edit code, add features, improve documentation, or experiment with new models.
   - Commit your changes regularly:
     ```bash
     git add .
     git commit -m "Describe your change"
     ```
   Follow commit [conventions](https://www.conventionalcommits.org/en/v1.0.0/) in your change message.

3. **Push Your Changes to GitHub**
   - Push your branch to your fork:
     ```bash
     git push origin my-feature-branch
     ```

4. **Showcase Your Work**
   - **Update the [ReadMe file](README.md)** to describe the new features you added.
   - Add your deployed Streamlit app url to the ReadMe file.
   - Screen record the Streamlit app
      - walk the viewer through overall app
      - demonstrate your own updates
   - Turn the recorded video to gif image and add to your ReadMe file.
   - For a detailed guide on how to write a job-winning ReadMe, read [this file](./Docs/6.How_to_Prepare_ReadMe.md).


5. **(Optional) Create a Pull Request**
   - If you think your changes could help others, open a pull request to the original repo to contribute back. For a step-by-step guide, see [How to Make a Pull Request](./Docs/5.How_to_Make_a_Pull_Request.md).
   - **This will improve your GitHub presence, which is publicly trackable by future employers**

### Present Your Project to Potential Hiring Managers
   - Make a LinkedIn post about your project, and the value you added.
   - See a detailed guide on how to write a catchy LinkedIn post [here](./Docs/7.Present_project_on_LinkedIn.md).
