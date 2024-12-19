import os
import zipfile
import subprocess
import pandas as pd
from datetime import date, timedelta, datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def get_latest_master():

    """ downloads latest price data and saves as master1.csv"""

    subprocess.run(["kaggle", "datasets", "download", "-d", "benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated", "-p", "data"])
    #!kaggle datasets download -d benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated -p data
    zip_file_path = '/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/data/s-and-p-500-with-dividends-and-splits-daily-updated.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/data')

    df = pd.read_csv('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/archive/sp500_stocks.csv')
    sp = df["Symbol"].unique()
    sp.sort()
    sp = list(sp)

    data = os.listdir('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/data')
    data.sort()
    data = list(data)

    # create master dataframe from single dfs
    df = pd.DataFrame()
    for i in data:
        if i.replace(".csv", "") in sp:
            df2 = pd.read_csv(f"data/{i}")
            df2["company"] = i.replace(".csv", "")
            df = pd.concat([df, df2], ignore_index=True)

    #df.to_csv("master1.csv")
    return df


def create_ML1_df(df):

    """ creates dataframe for machine learning prediction"""

    caps_df = pd.read_csv('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/caps_df.csv')
    sector_df = pd.read_csv('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/sector_df.csv')

    # add information of marketcap and (sub)sectors
    df = df.merge(caps_df, how="inner", on="company")
    df = df.merge(sector_df, how="inner", on="company")
    df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)

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
        #company["price_30d"] = company["close"].shift(-30)

        return company

    df2 = grouped_df.apply(ma_bb).dropna()

    # dropping not needed price features
    df3 = df2.copy()
    df3.drop(columns=["open", "high", "low"], inplace=True)

    df3.drop(columns="company", inplace=True)
    df3.reset_index(inplace=True)
    df3.set_index("date", inplace=True)

    df3.to_csv("ml1.csv")
    return df3


def create_ML1_df_training(df):

    """ creates dataframe for machine learning training"""

    # import data
    #df = pd.read_csv("master1.csv")
    caps_df = pd.read_csv('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/caps_df.csv')
    sector_df = pd.read_csv('/Users/jangfeller/Library/Mobile Documents/com~apple~CloudDocs/_Ironhack/projects/mid_project/app/utils/sector_df.csv')

    # add information of marketcap and (sub)sectors
    df = df.merge(caps_df, how="inner", on="company")
    df = df.merge(sector_df, how="inner", on="company")
    df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)

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

    #df3.to_csv("ml1_training.csv")

    return df3


def run_xgb(df4, df4_training):

    """ creates df with top 20 predicted gains """

    def XGB_train_real():

        # short cleaning
        df_x = df4_training.copy().reset_index()
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

        df_ga = df4.copy()
        df_ga.reset_index(inplace=True)
        df_ga["date"] = pd.to_datetime(df_ga["date"])
        gains_df = pd.DataFrame(columns=["company", "close", "prediction", "gain_predicted"])
        df_ga.drop(columns=["sector", "subsector"], inplace=True)
        df_ga.set_index("date", inplace=True)    
        df_ga.sort_index(inplace=True)

        yesterday = date.today() - timedelta(days=2)
        if not yesterday in df_ga:
            yesterday = date.today() - timedelta(days=3)
            if not yesterday in df_ga:
                yesterday = date.today() - timedelta(days=4)
                if not yesterday in df_ga:
                    yesterday = date.today() - timedelta(days=5)
                    if not yesterday in df_ga:
                        yesterday = date.today() - timedelta(days=6)
                        if not yesterday in df_ga:
                            yesterday = date.today() - timedelta(days=7)
                            if not yesterday in df_ga:
                                yesterday = date.today() - timedelta(days=8)
                                if not yesterday in df_ga:
                                    yesterday = date.today() - timedelta(days=9)

        xg_df = df_ga.loc[str(yesterday)]
        xg_df.sort_index(inplace=True)

        X = xg_df.drop(columns=["company"])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        pred_xgb = xgbr.predict(X_scaled)
        test = xg_df[["company", "close"]]
        test["prediction"] = pred_xgb
        test["gain_predicted"] = (test["prediction"] - test["close"]) / test["close"] * 100
        test.sort_values(by="gain_predicted", ascending=False, inplace=True)
        test.reset_index(inplace=True)
        test.drop(columns="date", inplace=True)
        test.set_index("company", inplace=True)

        return test

    

    xgbr = XGB_train_real()
    stocks = get_stocks()
    #stocks.to_csv("top_20.csv")
    return stocks



def plot_data(df, top):
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df2 = df.loc["2020":]
    df3 = df2.drop(columns=["ma_60", "rsi", "macd", "volume", "dividends", "stock_splits", "marketcap", "gain_14", "loss_14", "rsi_14", "ema_12", "ema_26", "rsi_lag_5", "rsi_lag_10"])
    df3.reset_index(inplace=True)

    future = datetime.today().date() + timedelta(days=30)
    top["date"] = pd.to_datetime(future)

    top.drop(columns=["close", "gain_predicted"], inplace=True)
    top.rename(columns={"prediction": "close"}, inplace=True)

    merged = pd.concat([df3, top], ignore_index=True)
    merged.sort_values(by="date", inplace=True)

    return merged