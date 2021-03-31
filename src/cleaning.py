import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import time

def merge_dataframes(df1, df2, df3):
    new_df = pd.merge(left=df1, right=df2, how="right")
    new_df = pd.merge(left=new_df, right=df3, how="left")
    return new_df

def drop_initial_unused_cols(df, lst_of_cols_to_drop):
    return df.drop(lst_of_cols_to_drop, axis=1)

def create_datetime_dummies(df):
    df["timestamp"]= pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    spring = (df["date"] >= pd.to_datetime("2016-01-01").date()) & (df["date"] < pd.to_datetime("2016-04-01").date())
    summer = (df["date"] >= pd.to_datetime("2016-04-01").date())& (df["date"] < pd.to_datetime("2016-07-01").date())
    fall = (df["date"] >= pd.to_datetime("2016-07-01").date())& (df["date"] < pd.to_datetime("2016-10-01").date())    
    winter = (df["date"] >= pd.to_datetime("2016-10-01").date())& (df["date"] <= pd.to_datetime("2016-12-31").date())
    df["date"][spring] = "q1"
    df["date"][summer] = "q2"
    df["date"][fall] = "q3"
    df["date"][winter] = "q4"
    date_dummies = pd.get_dummies(df["date"])
    df = df.join(other=date_dummies)
    df.drop(["date"], axis=1, inplace=True)

    df["time"] = df["timestamp"].dt.time
    early_morning = (df["time"] >= pd.to_datetime("00:00:00").time()) & (df["time"] < pd.to_datetime("06:00:00").time()) 
    morning = (df["time"] >= pd.to_datetime("06:00:00").time()) & (df["time"] < pd.to_datetime("12:00:00").time())
    afternoon = (df["time"] >= pd.to_datetime("12:00:00").time()) & (df["time"] < pd.to_datetime("18:00:00").time())
    evening = (df["time"] >= pd.to_datetime("18:00:00").time()) & (df["time"] <= pd.to_datetime("23:59:00").time())
    df["time"][early_morning] = "early_morning"
    df["time"][morning] = "morning"
    df["time"][afternoon] = "afternoon"
    df["time"][evening] = "evening"
    time_dummies = pd.get_dummies(df["time"])
    df = df.join(other=time_dummies)
    df.drop(["time"], axis=1, inplace=True)
    return df

def create_primary_usage_and_meter_dummies(df):
    meter_dummies = pd.get_dummies(df["meter"])
    df = df.join(other=meter_dummies)
    df = df.rename(columns={0:"electricity", 1:"chilledwater", 2:"steam", 3:"hotwater"})
    df.drop(["meter"], axis=1, inplace=True)

    usage_dummies = pd.get_dummies(df["primary_use"])
    df = df.join(other=usage_dummies)
    df.drop(["primary_use", "Religious worship"], axis=1, inplace=True)
    return df

def convert_site_one_meter_readings(df, to_kWh=True):
    if to_kWh:
        df["meter_reading"].loc[df["site_id"] == 0] = df.loc[df["site_id"] == 0, "meter_reading"].apply(lambda x: x*0.2931)
    else:
        df["meter_reading"].loc[df["site_id"] == 0] = df.loc[df["site_id"] == 0, "meter_reading"].apply(lambda x: x*3.4118)
    return df

def create_temp_mean_df(weather_df):
    weather_df_copy = weather_df.copy()
    weather_df_copy["timestamp"]= pd.to_datetime(weather_train_df["timestamp"])
    weather_df_copy["date"] = weather_df_copy["timestamp"].dt.date
    weather_df_copy.drop(["cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", "timestamp"], axis=1, inplace=True)
    temp_mean_df = weather_df_copy.groupby(by=["site_id", "date"]).mean()
    return temp_mean_df

def impute_temp_nans(df, temp_mean_df):
    start = time.time()
    for idx, temp in enumerate(df["air_temperature"]):
        if pd.isna(temp):
            year = df.iloc[idx, 3].year
            month = df.iloc[idx, 3].month
            day = df.iloc[idx, 3].day
            df.iloc[idx, 5] = temp_mean_df.loc[(df.iloc[idx, 0] , datetime.date(year, month, day))]["air_temperature"]
        if idx % 100000 == 0:
            checkpoint = time.time()
            print(f'{idx} 100,000 time:{checkpoint-start}')
    return df

def create_ref_col_and_drop_remaining_unused(df):
    df = df.drop(["timestamp", "site_id", "building_id"], axis=1)
    df["row_id"] = df.index
    return df


if __name__ == "__main__":
    metadata_df = pd.read_csv("../data/ashrae-energy-prediction/building_metadata.csv")
    weather_train_df = pd.read_csv("../data/ashrae-energy-prediction/weather_train.csv")
    train_df = pd.read_csv("../data/ashrae-energy-prediction/train.csv")
    sample_submission_df = pd.read_csv("../data/ashrae-energy-prediction/sample_submission.csv")
    weather_test_df = pd.read_csv("../data/ashrae-energy-prediction/weather_test.csv")
    test_df = pd.read_csv("../data/ashrae-energy-prediction/test.csv")

    combined_df = merge_dataframes(metadata_df, train_df, weather_train_df)

    lst_of_cols_to_drop = ["cloud_coverage", "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", "year_built", "floor_count"]
    combined_df = drop_initial_unused_cols( combined_df, lst_of_cols_to_drop)
    combined_df = create_datetime_dummies(combined_df)
    combined_df = create_primary_usage_and_meter_dummies(combined_df)
    combined_df = convert_site_one_meter_readings(combined_df, to_kWh=True)
   
    temp_mean_df = create_temp_mean_df(weather_train_df)
    
    combined_df = impute_temp_nans(combined_df, temp_mean_df)
    cleaned_df = create_ref_col_and_drop_remaining_unused(combined_df)
    cleaned_df.to_csv("../data/cleaned_df.csv")




    

    


