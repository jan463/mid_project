def run_xgb():

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
        Output: df with top 20 gainers
        """

        yesterday = date.today() - timedelta(days=2)
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

    # import data
    df4 = pd.read_csv("ml1.csv")
    df4_training = pd.read_csv("ml1_training.csv")

    xgbr = XGB_train_real()
    stocks = get_stocks()
    return stocks.head(20)