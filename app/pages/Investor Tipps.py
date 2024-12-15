import streamlit as st

import os
import sys
import zipfile
import subprocess
import pandas as pd
from datetime import date, timedelta
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

############## page setup #########

st.set_page_config(
    layout="wide", 
    page_title="Investing S&P 500",
    page_icon="📈", 
    initial_sidebar_state="expanded"
)

################# functions ##################

sys.path.append('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/')
from functions import get_latest_master, create_ML1_df, create_ML1_df_training, run_xgb

def get_data():
    step_1 = get_latest_master()
    print("data downloaded")
    ml1 = create_ML1_df(step_1)
    print("ML1 created")
    ml1_training = create_ML1_df_training(step_1)
    print("ML1 training created")
    tipps = run_xgb(ml1, ml1_training)
    print("tipps ready")
    return tipps




################ title ######################

st.title("Investor Page")
st.header("Investing in Stocks of the S&P 500")

st.divider()


############### body #################

if "tipps" not in st.session_state:
    st.session_state.tipps = None

if st.button("Download and process Data"):
    st.write("The Data is loading...")
    st.session_state.tipps = get_data()
    
st.markdown(body=":red[_This step may take several minutes_]")
st.write("")

st.markdown("""
Download the latest price data from all S&P 500 stocks to use as input for 
the machine learning model.  \n The model trains with this data and then outputs 
a table, showing predicted gains sorted in descending order.
""")

st.write("")
st.write("")

st.subheader("Stock Price Predictions")

if st.session_state.tipps is not None:
    st.dataframe(st.session_state.tipps)
