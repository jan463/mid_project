
import pandas as pd
import os
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
import scipy.stats as stats
import zipfile
from sklearn.preprocessing import MinMaxScaler 
import xgboost as xgb
from datetime import date, timedelta

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import subprocess

import warnings
warnings.filterwarnings("ignore")

# functions

def get_latest_master():
    subprocess.run(["kaggle", "datasets", "download", "-d", "benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated", "-p", "data"])
    #!kaggle datasets download -d benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated -p data
    zip_file_path = "data/s-and-p-500-with-dividends-and-splits-daily-updated.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("data")

    df = pd.read_csv("archive/sp500_stocks.csv")
    sp = df["Symbol"].unique()
    sp.sort()
    sp = list(sp)

    data = os.listdir("data")
    data.sort()
    data = list(data)

    # create master dataframe from single dfs
    df = pd.DataFrame()
    for i in data:
        if i.replace(".csv", "") in sp:
            df2 = pd.read_csv(f"data/{i}")
            df2["company"] = i.replace(".csv", "")
            df = pd.concat([df, df2], ignore_index=True)

    df.to_csv("master1.csv")


get_latest_master()

# script creates dataframe for ML
import create_ML1_df
ml1 = create_ML1_df.create_ML1_df()
ml1.to_csv("ml1")

import create_ML1_df_training
ml2 = create_ML1_df_training.create_ML1_df_training()
ml2.to_csv("ml1_training")
