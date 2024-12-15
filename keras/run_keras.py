def keras():

        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from datetime import date, timedelta

        from sklearn.model_selection import train_test_split 
        from sklearn.preprocessing import MinMaxScaler 

        import xgboost as xgb
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        import keras
        from keras.models import load_model


        # import data
        df5 = pd.read_csv("ml1.csv")

        #df5.drop(columns=["company.1"], inplace=True)

        # short cleaning
        df5.set_index("date", inplace=True)
        df5.sort_index(inplace=True)

        yesterday = date.today() - timedelta(days=2)
        df5 = df5.loc[str(yesterday)]

        df5["company"] = df5["company"].astype("category")
        df5["company"] = df5["company"].cat.codes
        df5["sector"] = df5["sector"].astype("category")
        df5["sector"] = df5["sector"].cat.codes
        df5["subsector"] = df5["subsector"].astype("category")
        df5["subsector"] = df5["subsector"].cat.codes

        """
        # create training and test data
        training = df5.loc[:"2023"]
        test = df5.loc["2023":]

        X_training = training.drop(columns="price_30d")
        X_test = test.drop(columns="price_30d")
        y_training = training["price_30d"]
        y_test = test["price_30d"]



        # separate categorical features
        company_training = X_training["company"]
        category_training = X_training["sector"]
        subcategory_training = X_training["subsector"]
        X_training.drop(columns=["company", "sector", "subsector"], inplace=True)

        """

        X_test = df5.copy()
        #X_test.drop(columns="price_30d", inplace=True)

        company_test = X_test["company"]
        category_test = X_test["sector"]
        subcategory_test = X_test["subsector"]
        X_test.drop(columns=["company", "sector", "subsector"], inplace=True)

        scaler = MinMaxScaler()
        #X_training_scaled = scaler.fit_transform(X_training)
        X_test_scaled = scaler.fit_transform(X_test)

        from keras.losses import mean_squared_error

        model = keras.models.load_model("model_1.h5",
        custom_objects={"mse": mean_squared_error})

        k_predictions = model.predict([X_test_scaled, company_test, category_test, subcategory_test])

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
                pred_keras = model.predict([X_test_scaled, company_test, category_test, subcategory_test])
                test = xg_df[["close", "company"]]
                test["prediction"] = pred_keras
                test["gain_predicted"] = (test["prediction"] - test["close"]) / test["close"] * 100
                test.sort_values(by="gain_predicted", ascending=False, inplace=True)
                #gain = test.head(10)["gain_real"].mean()

                return test


        # import data
        df4 = pd.read_csv("ml1.csv")
        df4_training = pd.read_csv("ml1_training.csv")

        stocks = get_stocks()
        return stocks