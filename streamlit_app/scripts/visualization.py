import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for plot styling
import networkx as nx

sns.set_theme(style="darkgrid")  # Set seaborn theme to whitegrid
sns.set_style(style="dark")
plt.rcParams["axes.edgecolor"] = "0.5"  # Set axes edge color
plt.rcParams["axes.linewidth"] = 1.5  # Set axes line width
def plot_bagging_circle():
    G = nx.DiGraph()  # Create a directed graph

    G.add_node("Data", pos=(1, 3), color="blue")  # Add data node
    for i, label in enumerate(
        ["Subset 1", "Subset 2", "Subset 3"]
    ):  # Loop over subsets
        G.add_node(label, pos=(2, 5 - i * 2), color="green")  # Add subset node
        G.add_edge("Data", label)  # Connect data to subset

        G.add_node(f"Model {i+1}", pos=(3, 5 - i * 2), color="red")  # Add model node
        G.add_edge(label, f"Model {i+1}")  # Connect subset to model

    G.add_node(
        "Aggregated\nPrediction", pos=(4, 3), color="purple"
    )  # Add aggregation node
    for i in range(3):  # Connect models to aggregation
        G.add_edge(f"Model {i+1}", "Aggregated\nPrediction")

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(12, 7))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=4000,
        node_color=color,
        font_size=10,
        font_weight="bold",
        edge_color="0.4",
        width=1.25,
        alpha=0.6,
        arrowsize=20,
    )  # Draw the graph

    plt.title(
        "Bagging (Bootstrap Aggregating)", fontsize=14, fontweight="bold"
    )  # Set title
    #plt.show()  # Show plot
    return fig


def plot_boosting_circle():
    G = nx.DiGraph()  # Create a directed graph

    prev_model = None  # Track previous model
    for i in range(4):  # Loop over models
        current_model = f"Model {i+1}"  # Model name
        G.add_node(current_model, pos=(i + 1, 3), color="red")  # Add model node

        if prev_model:  # If not first model
            G.add_edge(prev_model, current_model)  # Connect previous to current
        prev_model = current_model  # Update previous model

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(12, 7))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=4000,
        node_color=color,
        font_size=10,
        font_weight="bold",
        edge_color="0.4",
        width=1.25,
        alpha=0.6,
        arrowsize=20,
    )  # Draw the graph

    plt.title(
        "Boosting: Sequential Model Training", fontsize=14, fontweight="bold"
    )  # Set title
    #plt.show()  # Show plot
    return fig
def plot_bagging_rect():
    G = nx.DiGraph()  # Create a directed graph

    G.add_node("Data", pos=(1, 3), color="#1f78b4")  # Add data node
    for i, label in enumerate(
        ["Subset 1", "Subset 2", "Subset 3"]
    ):  # Loop over subsets
        G.add_node(label, pos=(2, 5 - i * 2), color="#33a02c")  # Add subset node
        G.add_edge("Data", label)  # Connect data to subset

        G.add_node(
            f"Model {i+1}", pos=(3, 5 - i * 2), color="#e31a1c"
        )  # Add model node
        G.add_edge(label, f"Model {i+1}")  # Connect subset to model

    G.add_node(
        "Aggregated\nPrediction", pos=(4.5, 3), color="#6a3d9a"
    )  # Add aggregation node
    for i in range(3):  # Connect models to aggregation
        G.add_edge(f"Model {i+1}", "Aggregated\nPrediction")

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(14, 8))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=5000,
        node_color=color,
        font_size=11,
        font_weight="bold",
        edge_color="0.4",
        width=1.5,
        alpha=0.7,
        arrowsize=25,
        node_shape="s",
        edgecolors="#000000",
        linewidths=1.5,
    )  # Draw graph

    plt.title(
        "Bagging (Bootstrap Aggregating)", fontsize=16, fontweight="bold", pad=20
    )  # Set title
    # plt.savefig(
    #     "output/bagging_plot.png", dpi=300, bbox_inches="tight"
    # )  # Save plot to file
    # print("Saved bagging plot to bagging_plot.png")  # Print confirmation
    return fig


def plot_boosting_rect():
    G = nx.DiGraph()  # Create a directed graph

    prev_model = None  # Track previous model
    for i in range(4):  # Loop over models
        current_model = f"Model {i+1}"  # Model name
        G.add_node(
            current_model, pos=(i * 1.5 + 1, 3), color="#e31a1c"
        )  # Add model node

        if prev_model:  # If not first model
            G.add_edge(prev_model, current_model)  # Connect previous to current
        prev_model = current_model  # Update previous model

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(14, 8))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=5000,
        node_color=color,
        font_size=12,
        font_weight="bold",
        edge_color="0.4",
        width=1.5,
        alpha=0.7,
        arrowsize=25,
        node_shape="s",
        edgecolors="#000000",
        linewidths=1.5,
    )  # Draw graph

    plt.title(
        "Boosting: Sequential Model Training", fontsize=16, fontweight="bold", pad=20
    )  # Set title
    # plt.savefig(
    #     "output/boosting_plot.png", dpi=300, bbox_inches="tight"
    # )  # Save plot to file
    # print("Saved boosting plot to boosting_plot.png")  # Print confirmation
    return fig
def plot_stacking():
    G = nx.DiGraph()  # Create a directed graph

    # Base models
    for i, label in enumerate(
        ["Model 1", "Model 2", "Model 3"]
    ):  # Loop over base models
        G.add_node(label, pos=(1, 5 - i * 2), color="#e31a1c")  # Add base model node

    # Predictions from base models
    for i, label in enumerate(["Pred 1", "Pred 2", "Pred 3"]):  # Loop over predictions
        G.add_node(label, pos=(2.5, 5 - i * 2), color="#33a02c")  # Add prediction node
        G.add_edge(f"Model {i+1}", label)  # Connect model to prediction

    # Meta-model making final prediction based on base model predictions
    G.add_node("Meta-Model", pos=(4, 3), color="#1f78b4")  # Add meta-model node
    for i in range(3):  # Connect predictions to meta-model
        G.add_edge(f"Pred {i+1}", "Meta-Model")

    G.add_node(
        "Final\nPrediction", pos=(5.5, 3), color="#6a3d9a"
    )  # Add final prediction node
    G.add_edge(
        "Meta-Model", "Final\nPrediction"
    )  # Connect meta-model to final prediction

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(16, 8))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=5000,
        node_color=color,
        font_size=12,
        font_weight="bold",
        edge_color="0.4",
        width=1.5,
        alpha=0.7,
        arrowsize=25,
        node_shape="s",
        edgecolors="#000000",
        linewidths=1.5,
    )  # Draw graph

    plt.title(
        "Stacking (Stacked Generalization)", fontsize=16, fontweight="bold", pad=20
    )  # Set title
    # plt.savefig(
    #     "output/stacking_plot.png", dpi=300, bbox_inches="tight"
    # )  # Save plot to file
    # print("Saved stacking plot to stacking_plot.png")  # Print confirmation
    return fig
def plot_voting():
    G = nx.DiGraph()  # Create a directed graph

    # Base models making their predictions
    for i, label in enumerate(
        ["Model 1", "Model 2", "Model 3"]
    ):  # Loop over base models
        G.add_node(label, pos=(1, 5 - i * 2), color="#e31a1c")  # Add base model node

    # Voting mechanism
    G.add_node(
        "Voting\nMechanism", pos=(3, 3), color="#1f78b4"
    )  # Add voting mechanism node
    for i in range(3):  # Connect models to voting mechanism
        G.add_edge(f"Model {i+1}", "Voting\nMechanism")

    # Final Decision after Voting
    G.add_node(
        "Final\nDecision", pos=(5, 3), color="#6a3d9a"
    )  # Add final decision node
    G.add_edge(
        "Voting\nMechanism", "Final\nDecision"
    )  # Connect voting mechanism to final decision

    pos = nx.get_node_attributes(G, "pos")  # Get node positions
    color = [G.nodes[node]["color"] for node in G.nodes]  # Get node colors

    fig=plt.figure(figsize=(16, 8))  # Set figure size
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=5000,
        node_color=color,
        font_size=11,
        font_weight="bold",
        edge_color="0.4",
        width=1.5,
        alpha=0.7,
        arrowsize=25,
        node_shape="s",
        edgecolors="#000000",
        linewidths=1.5,
    )  # Draw graph

    plt.title(
        "Voting Mechanism in Ensemble Learning", fontsize=16, fontweight="bold", pad=20
    )  # Set title
    # plt.savefig(
    #     "output/voting_plot.png", dpi=300, bbox_inches="tight"
    # )  # Save plot to file
    #print("Saved voting plot to voting_plot.png")  # Print confirmation
    return fig


def plot_feature_importance(importance, names, model_name):
    # Convert to list if needed
    if not isinstance(names, list):
        names = names.tolist()

    # Create feature importance pairs
    feature_importance = list(zip(names, importance))
    sorted_feature_importance = sorted(
        feature_importance, key=lambda x: x[1], reverse=True
    )  # Sort by importance
    names_sorted, vals_sorted = zip(*sorted_feature_importance)  # Unpack sorted pairs

    fig=plt.figure(figsize=(12, 6))  # Set figure size
    plt.barh(
        range(len(names_sorted)), vals_sorted, align="center"
    )  # Horizontal bar plot
    plt.yticks(range(len(names_sorted)), names_sorted)  # Set y-tick labels
    plt.title(model_name + " - Feature Importance")  # Title
    plt.xlabel("Importance Value")  # X-axis label
    plt.ylabel("Features")  # Y-axis label
    plt.tight_layout()  # Adjust layout
    #plt.show()  # Display plot
    return fig


def plot_predictions(model_name, predictions,X_test,y_test):
    fig=plt.figure(figsize=(14, 6))  # Set figure size
    plt.plot(
        X_test.index, y_test, label="True Values", color="blue"
    )  # Plot true values
    plt.plot(
        X_test.index, predictions, label=f"{model_name} Predictions", linestyle="--"
    )  # Plot predicted values

    plt.title(f"{model_name} - True vs. Predicted BTC Prices Over Time")  # Title
    plt.xlabel("Date")  # X-axis label
    plt.ylabel("BTC Price (USDT)")  # Y-axis label
    plt.legend()  # Show legend
    #plt.show()  # Display plot
    return fig