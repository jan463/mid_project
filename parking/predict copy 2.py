def predict():
    
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

    import warnings
    warnings.filterwarnings("ignore")

    # functions

    def get_latest_master():
        !kaggle datasets download -d benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated -p data
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



    def XGB_train_real():

        # short cleaning
        df_x = df4_training.copy()
        df_x.drop(columns=["company", "sector", "subsector"], inplace=True)
        df_x.set_index("date", inplace=True)
        df_x.sort_index(inplace=True)

        # training with data only until training_end
        xg_df = df_x.copy()

        # X y split
        X = xg_df.drop(columns="price_30d")
        y = xg_df["price_30d"]

        # normalization
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # train model
        xgbr = xgb.XGBRFRegressor()
        xgbr.fit(X_scaled, y)

        return xgbr



    def get_stocks():

        """
        Output: df with top ten gainers
        """

        yesterday = date.today() - timedelta(days=1)
        df_ga = df4.copy()
        gains_df = pd.DataFrame(columns=["company", "close", "prediction", "gain_predicted"])
        df_ga.drop(columns=["sector", "subsector"], inplace=True)
        df_ga.set_index("date", inplace=True)    
        df_ga.sort_index(inplace=True)

        xg_df = df_ga.loc[str(yesterday)]
        xg_df.sort_index(inplace=True)

        X = xg_df.drop(columns=["company"])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        pred_xgb = xgbr.predict(X_scaled)
        test = xg_df[["close", "company"]]
        test["prediction"] = pred_xgb
        test["gain_predicted"] = (test["prediction"] - test["close"]) / test["close"] * 100
        test.sort_values(by="gain_predicted", ascending=False, inplace=True)
        #gain = test.head(10)["gain_real"].mean()

        return test


    get_latest_master()

    # script creates dataframe for ML
    import create_ML1_df
    import create_ML1_df_training

    # import data
    df4 = pd.read_csv("ml1.csv")
    df4_training = pd.read_csv("ml1_training.csv")

    xgbr = XGB_train_real()
    stocks = get_stocks()
    return stocks.head(20)