# -*- coding: utf-8 -*-  # Specifies the encoding of the file

"""
"""

import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for plot styling
import networkx as nx  # Import networkx for graph visualization
import streamlit as st
from scripts.visualization import (plot_boosting_circle,plot_bagging_circle,
                                   plot_boosting_rect,plot_bagging_rect,
                                   plot_stacking,plot_voting)

# Setting a theme and style
sns.set_theme(style="darkgrid")  # Set seaborn theme to whitegrid
sns.set_palette(palette=["blue","red","brown"])
sns.set_style(style="dark")
plt.rcParams["axes.edgecolor"] = "0.8"  # Set axes edge color
plt.rcParams["axes.linewidth"] = 4.5  # Set axes line width
st.subheader("Lets Review bagging and boosting techniques")
st.divider()
st.header("bagging and boosting".upper())
st.subheader("Ensemble learning diagrams")
st.markdown("We will first visualizes common ensemble strategies with \
            clean diagrams using NetworkX and Matplotlib.")
with st.container(border=True):
    tab1,tab2=st.tabs(["bagging preview", "boosting preview"])
# Execute the plotting functions
with tab1:
    with st.expander(label="bagging circle".upper(),expanded=True):
        st.pyplot(plot_bagging_circle())  # Plot bagging diagram
with tab2:
    with st.expander(label="boosting circle".upper(),expanded=True):
        st.pyplot(plot_boosting_circle())  # Plot boosting diagram

# Execute the plotting functions
with tab1:
    with st.expander(label="bagging rectangle",expanded=True):
        st.pyplot(plot_bagging_rect())  # Plot bagging diagram
with tab2:
    with st.expander(label="boosting rectangle",expanded=True):
        st.pyplot(plot_boosting_rect())  # Plot boosting diagram

# Execute the plotting function
st.divider()
st.title("Stacking method")
st.subheader("Stacking")
st.markdown("Defines `plot_stacking()` showing base models → predictions → meta-model → final prediction"\
            "; then renders the diagram.")
with st.expander("stacking",expanded=True):
    with st.container(border=True,key="stacking"):
        st.pyplot(plot_stacking())  # Plot stacking diagram

# Execute the plotting function
st.divider()
st.title("Voting method")
st.subheader("Voting")
st.markdown("Defines `plot_voting()` where three base models feed a voting" \
    " mechanism;then renders the final decision.")
with st.expander("voting",expanded=True):
    with st.container(border=True):
        st.pyplot(plot_voting())  # Plot voting diagram

st.markdown("Notice how the voting mechanism aggregates predictions \n \
            from multiple base models to make a final decision,\
             enhancing overall model robustness and accuracy.")