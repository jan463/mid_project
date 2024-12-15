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




sys.path.append('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/')
from functions import get_latest_master, create_ML1_df, create_ML1_df_training, run_xgb

st.title("Investor Page")
st.header("Investing in Stocks of the S&P 500")

st.divider()

def get_data():
    step_1 = get_latest_master()
    ml1 = create_ML1_df(step_1)
    ml1_training = create_ML1_df_training(step_1)
    tipps = run_xgb(ml1, ml1_training)
    return tipps

def download():
    get_latest_master()




if st.button("Download and process Data"):
    st.write("The Data is loading...")
    get_data()

st.write("Press Button to download latest Price Data")

top_20 = pd.read_csv("/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/pages/top_20.csv")
st.dataframe(top_20)

if st.button("Download latest Price Data"):
    st.write("The Data is loading...")
    get_data()