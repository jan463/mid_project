def create_ML1_df_training():

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from sklearn.model_selection import train_test_split 
    from sklearn.preprocessing import MinMaxScaler 

    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # import data
    df = pd.read_csv("master1.csv")
    caps_df = pd.read_csv("caps_df.csv")
    sector_df = pd.read_csv("sector_df.csv")

    # add information of marketcap and (sub)sectors
    df = df.merge(caps_df, how="inner", on="company")
    df = df.merge(sector_df, how="inner", on="company")
    df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0"], inplace=True)

    # data cleaning
    df.columns = [columns.lower().replace(" ", "_") for columns in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["marketcap"] = df["marketcap"].map(lambda x: x.replace("B", "").replace(",", ""))
    df["marketcap"] = df["marketcap"].map(lambda x: pd.to_numeric(x))

    # moving average, bollinger bands, price change, RSI
    grouped_df = df.groupby("company")

    def ma_bb(company):
        company = company.sort_values(by="date").set_index("date")

        # indicators
        company["ma_20"] = company["close"].rolling(window=20).mean()
        company["ma_60"] = company["close"].rolling(window=60).mean()
        company["bb_lower"] = company["close"].rolling(window=20).mean() - company["close"].rolling(window=20).std()*2
        company["bb_upper"] = company["close"].rolling(window=20).mean() + company["close"].rolling(window=20).std()*2

        company["price_change"] = company["close"].diff()

        company["gain_14"] = company["price_change"].clip(lower=0).rolling(window=14).mean()
        company["loss_14"] = company["price_change"].clip(upper=0).rolling(window=14).mean()

        company["rsi"] = 100 - (100 / (1 + (company["gain_14"] / abs(company["loss_14"]))))
        company["rsi_14"] = company["rsi"].rolling(window=14).mean()

        company["ema_12"] = company["close"].ewm(span=12, adjust=False).mean()
        company["ema_26"] = company["close"].ewm(span=26, adjust=False).mean()
        company["macd"] = (company["ema_12"] - company["ema_26"]).ewm(span=9, adjust=False).mean()


        # lagged indicators
        company["rsi_lag_5"] = company["rsi_14"].shift(5)
        company["rsi_lag_10"] = company["rsi_14"].shift(10)

        # target
        company["price_30d"] = company["close"].shift(-30)

        return company

    df2 = grouped_df.apply(ma_bb).dropna()

    # dropping not needed price features
    df3 = df2.copy()
    df3.drop(columns=["open", "high", "low"], inplace=True)

    df3.drop(columns="company", inplace=True)
    df3.reset_index(inplace=True)
    df3.set_index("date", inplace=True)

    df3.to_csv("ml1_training.csv")

    return df3
