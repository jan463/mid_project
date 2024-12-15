import streamlit as st

import os
import sys
import zipfile
import subprocess
import pandas as pd
from datetime import datetime, date, timedelta
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
from functions import get_latest_master, create_ML1_df, create_ML1_df_training, run_xgb, plot_data

def get_data():
    step_1 = get_latest_master()
    print("data downloaded")
    ml1 = create_ML1_df(step_1)
    ml1.to_csv("ml1.csv")
    print("ML1 created")
    ml1_training = create_ML1_df_training(step_1)
    print("ML1 training created")
    tipps = run_xgb(ml1, ml1_training)
    tipps.to_csv("top_20.csv")
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

st.write("")
st.write("")

################ plot #############
st.subheader("Visualize the Data")

df = pd.read_csv("ml1.csv")
top = pd.read_csv("top_20.csv")


merged = plot_data(df, top)
merged.drop(columns="subsector", inplace=True)
merged["date"] = pd.to_datetime(merged["date"]).dt.date
merged.sort_values(by="date", inplace=True)

#st.dataframe(merged)

company = st.selectbox("Select a Company: ", merged["company"].unique())
#filtered_df = merged[merged["company"] == company]


#st.dataframe(filtered_df)


############ filter for date ###########

zooms = ["6 Months", "1 Year", "3 Years"]
selected_zoom = st.selectbox("Select the Timeframe: ", zooms, index=2)

end_date = merged["date"].max()
if selected_zoom == "6 Months":
    start_date = end_date - timedelta(days=182)
elif selected_zoom == "1 Year":
    start_date = end_date - timedelta(days=365)
elif selected_zoom == "3 Years":
    start_date = end_date - timedelta(days=3*365)

filtered_df = merged[(merged["company"] == company) & (merged["date"] >= start_date)]

################ actual plot #################

fig = px.line(filtered_df, x='date', y=['bb_lower', 'ma_20', 'bb_upper', 'close'],
              labels={'value': 'Values', 'date': 'Date'},
              title='Moving Averages for Sample Company')

# Add horizontal lines using Plotly
fig.add_hline(y=70, line_dash="dash", line_color="gray")
fig.add_hline(y=30, line_dash="dash", line_color="gray")

# Update layout to improve the aesthetics
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Values',
    legend_title='Measure',
    legend=dict(font_size=12),
    title_font=dict(size=16, family='Arial', color='black')
)

# Display the interactive plot in Streamlit
st.plotly_chart(fig)