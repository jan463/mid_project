import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import requests
import os
from dotenv import load_dotenv

############## page setup #########

st.set_page_config(
    page_title="S&P 500",
    layout="wide", 
    page_icon="📈", 
    initial_sidebar_state="expanded"
)


############### title ##############

st.image("titlepage.png")


st.header("This application provides general information and recent trends in the S&P 500.")
st.subheader("Overview: Average Price Data of S&P500 normalized since 2010")



########## average plot ###########
mean_2010 = pd.read_csv("mean_2010.csv")
mean_2010["date"] = pd.to_datetime(mean_2010["date"], utc=True)
fig, ax = plt.subplots(figsize=(14, 6))  
sns.lineplot(data=mean_2010, x="date", y="close")

ax.set_facecolor("#eaeaea")
ax.grid(True, linestyle="--", alpha=0.7, color="black")
ax.set_title("Normalized Mean S&P 500", fontsize=20, color="black")
ax.set_xlabel("Date", fontsize=15, color="black")
ax.set_ylabel("Closing Price", fontsize=15, color="black")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xticks(color="black")
plt.yticks(color="black")

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

############ text about sectors ############
st.write("""
The S&P 500 Index represents the economic heartbeat of the United States, encompassing a 
diverse array of companies across multiple sectors. Analyzing these sectors offers valuable 
insights into their individual and collective contributions to market performance. This plot 
provides a visual representation of the S&P 500's sector composition, highlighting key players 
and sector weights. By understanding these dynamics, investors can better assess market trends, 
identify opportunities for diversification, and gauge economic health. Whether you're tracking 
technologys impact, the stability of utilities, or the ebb and flow of consumer behavior, 
this sector analysis serves as a pivotal tool in your market evaluation toolkit.
""")

st.write("")
st.write("")

############# news stream #################

st.header("Finance News Stream")

load_dotenv()
api_key = os.getenv("api_key")

url = 'https://newsapi.org/v2/everything'
params = {
    'q': 'finance',
    'sortBy': 'publishedAt',
    'apiKey': api_key
}
response = requests.get(url, params=params)
data = response.json()


for i in range(21):
    st.subheader(data["articles"][i]["title"])
    st.write(data["articles"][i]["description"])
    st.markdown(f"[Read More]({data['articles'][i]['url']})")
    st.write("published at ",pd.to_datetime(data["articles"][i]["publishedAt"]).tz_localize(None))




